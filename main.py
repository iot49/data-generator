#!/usr/bin/env python3

# ruff: noqa: F403, F405
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportAny=false

import argparse
import random
from pathlib import Path
from typing import cast as typing_cast, Any

import matplotlib.pyplot as plt
import torch
import numpy as np
import onnxruntime as ort
from fastai.vision.all import *

from r49 import R49DataLoaders, R49Dataset

# Using explicit types for better readability and type checking where possible
LABELS = ["track", "train", "other"]
SIZE = 96
DPT = 30
VALID_PCT = 0.5
DEFAULT_TUNE_STEPS = 300
DEFAULT_BATCH_SIZE = 64

class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        
    def forward(self, x):
        return (x - self.mean) / self.std

def parse_args():
    parser = argparse.ArgumentParser(description="Train a ResNet model on R49 dataset.")
    parser.add_argument("--model-name", type=str, default="resnet18", help="Name of the model to train")
    parser.add_argument("--dataset-dir", type=Path, default=Path("../datasets/train-track/r49"), help="Directory containing the dataset")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory to save the trained model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_TUNE_STEPS, help="Number of fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--use-timm", action="store_true", help="Use timm model instead of standard torchvision/fastai model")
    parser.add_argument("--train", action="store_true", help="Run training pipeline")
    parser.add_argument("--verify", action="store_true", help="Run verification on exported model")
    return parser.parse_args()


def get_architecture(model_name: str):
    """
    Resolves the model architecture from a string name.
    """
    # Check if it's available in the current namespace (e.g. imported from fastai)
    if model_name in globals():
        return globals()[model_name]
    
    # Check if it's available in torchvision.models
    import torchvision.models as tvm
    if hasattr(tvm, model_name):
        return getattr(tvm, model_name)
        
    raise ValueError(f"Architecture '{model_name}' not found. Please ensure it is a valid torchvision model or fastai architecture.")


def run_inference(dl, session: ort.InferenceSession):
    correct = 0
    total = 0
    
    input_name = session.get_inputs()[0].name
    expected_batch_size = session.get_inputs()[0].shape[0]

    # Check if model currently supports dynamic batch (usually represented as 'batch_size' or generic string/None)
    # If fixed to 1, we must iterate one by one or create a new DL with bs=1
    is_fixed_batch_1 = expected_batch_size == 1
    
    if is_fixed_batch_1 and dl.bs != 1:
        print(f"Warning: Model expects batch size 1, but DL has bs={dl.bs}. Adjusting iteration...")
    
    for xb, yb in dl:
        # Convert to numpy
        # FastAI dataloaders return TensorImage, we need standard float array
        imgs = xb.cpu().numpy()
        labels = yb.cpu().numpy()
        
        # If model expects batch size 1, we loop over the batch
        if is_fixed_batch_1:
            for i in range(len(imgs)):
                img = imgs[i:i+1] # Keep 4D shape (1, C, H, W)
                
                # Run inference
                ort_inputs = {input_name: img}
                ort_outs = session.run(None, ort_inputs)
                
                # Output is (1, num_classes)
                probs = typing_cast(Any, ort_outs[0])
                pred = np.argmax(probs, axis=1)[0]
                
                if pred == labels[i]:
                    correct += 1
                total += 1
        else:
            # Run inference on full batch
            ort_inputs = {input_name: imgs}
            ort_outs = session.run(None, ort_inputs)
            
            probs = typing_cast(Any, ort_outs[0])
            preds = np.argmax(probs, axis=1)
            
            correct += np.sum(preds == labels)
            total += len(labels)
            
    return correct / total if total > 0 else 0.0


def main():
    args = parse_args()
    
    # deterministic
    random.seed(42)

    if not args.train and not args.verify:
        print("Warning: Neither --train nor --verify specified. Use --help for usage information.")
        return

    print(f"Dataset Dir: {args.dataset_dir}")
    print(f"Model Name: {args.model_name}")
    print(f"Use Timm: {args.use_timm}")
    
    # data
    ds = R49Dataset(args.dataset_dir.rglob("**/*.r49"), dpt=DPT, size=int(1.5*SIZE), labels=LABELS)
    dls = R49DataLoaders.from_dataset(ds, valid_pct=VALID_PCT, crop_size=SIZE, bs=args.batch_size)
    
    # Note: fastai's vision_learner automatically adds normalization to the DataLoaders 
    # when using a pretrained model (defaults to ImageNet stats), so the training data IS normalized.
    # The normalization_layer in the final model ensures the exported model also handles normalization.

    if args.train:
        # Resolution: fastai's string support requires 'timm'. 
        if args.use_timm:
            arch = args.model_name
        else:
            # Since timm might not be installed, we manually resolve the architecture.
            arch = get_architecture(args.model_name)

        learn = vision_learner(dls, arch, metrics=error_rate)
        
        model_path = args.model_dir / f"{args.model_name}.pth"
        if model_path.exists():
            print(f"Loading saved model parameters from {model_path}")
            try:
                saved_model = torch.load(model_path, map_location=dls.device)
                learn.model.load_state_dict(saved_model.state_dict())
            except Exception as e:
                print(f"Warning: Failed to load saved model parameters: {e}")

        learn.fine_tune(args.epochs)

        # save model 
        args.model_dir.mkdir(parents=True, exist_ok=True)
        # Use config-based names for output
        torch.save(learn.model, args.model_dir / f"{args.model_name}.pth")

        pytorch_model = learn.model.eval()

        # define layers
        softmax_layer = torch.nn.Softmax(dim=1)
        normalization_layer = Normalization(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # assemble final model
        final_model = torch.nn.Sequential(
            normalization_layer,
            pytorch_model,
            softmax_layer
        )

        # Move entire final model to CPU
        final_model = final_model.eval().cpu()

        torch.onnx.export(
            final_model, 
            torch.randn(1, 3, SIZE, SIZE), args.model_dir / f"{args.model_name}.onnx", 
            do_constant_folding=True, 
            export_params=True,     
            input_names=[f'image_1_3_{SIZE}_{SIZE}'],
            output_names=['output'],
            opset_version=18,
            dynamo=False)

        print(final_model)

        interp = ClassificationInterpretation.from_learner(learn)
        interp.plot_confusion_matrix()
        interp.plot_top_losses(9, figsize=(9, 9))

        learn.show_results()
        plt.show()

    if args.verify:
        onnx_path = args.model_dir / f"{args.model_name}.onnx"
        print(f"Loading model from {onnx_path}...")

        try:
            if not onnx_path.exists():
                print(f"Error: {onnx_path} does not exist. Please run with --train first.")
            else:
                ort_session = ort.InferenceSession(str(onnx_path))
                
                # Validate on Training Set
                # dls.train is shuffled by default. Create a new one.
                print("Running inference on Training Set...")
                train_dl = dls.test_dl(dls.train_ds.items, with_labels=True, bs=1)
                train_acc = run_inference(train_dl, ort_session)
                print(f"Train Accuracy: {train_acc:.4f}")
                
                # Validate on Validation Set
                print("Running inference on Validation Set...")
                valid_dl = dls.test_dl(dls.valid_ds.items, with_labels=True, bs=1)
                valid_acc = run_inference(valid_dl, ort_session)
                print(f"Valid Accuracy: {valid_acc:.4f}")

        except Exception as e:
            print(f"An error occurred during inference: {e}")

if __name__ == "__main__":
    main()
