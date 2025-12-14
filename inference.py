#!/usr/bin/env python3

# ruff: noqa: F403, F405
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownMemberType=false
# pyright: reportAny=false

from typing import cast as typing_cast, Any

from pathlib import Path
import random
import matplotlib.pyplot as plt

from fastai.vision.all import *
import torch

from r49 import R49Dataset, R49DataLoaders

LABELS = ["track", "train", "other"]
MODEL_NAME = "resnet18"
MODEL_DIR = Path("models")
TUNE_STEPS = 3

R49_DIR = Path("../datasets/train-track/r49")

SIZE = 96
DB_SIZE = int(1.5*SIZE)
DPT = 30

VALID_PCT = 0.5

# deterministic
random.seed(42)

# data
ds = R49Dataset(R49_DIR.rglob("**/*.r49"), dpt=DPT, size=int(1.5*SIZE), labels=LABELS)
dls = R49DataLoaders.from_dataset(ds, valid_pct=VALID_PCT, crop_size=SIZE, bs=1)

# test
# TODO: load MODEL_DIR / f"{MODEL_NAME}.onnx" and test over the entire train and valid sets
import onnxruntime as ort
import numpy as np

def run_inference(dl, session: ort.InferenceSession):
    correct = 0
    total = 0
    
    input_name = session.get_inputs()[0].name
    expected_batch_size = session.get_inputs()[0].shape[0]

    # Check if model currently supports dynamic batch (usually represented as 'batch_size' or generic string/None)
    # If fixed to 1, we must iterate one by one or create a new DL with bs=1
    is_fixed_batch_1 = expected_batch_size == 1
    
    # If we are strictly bound to batch size 1 but DL has more, we might need to adjust strategy.
    # For now, let's assume if it is fixed 1, we will process item by item or ensure DL is bs=1.
    # But DL is already created.
    
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

onnx_path = MODEL_DIR / f"{MODEL_NAME}.onnx"
print(f"Loading model from {onnx_path}...")

try:
    if not onnx_path.exists():
        print(f"Error: {onnx_path} does not exist. Please run train.py first.")
    else:
        ort_session = ort.InferenceSession(str(onnx_path))
        
        # Validate on Training Set (no shuffle for consistency, though accuracy implies average)
        # dls.train is shuffled by default. Create a new one.
        train_dl = dls.test_dl(dls.train_ds.items, with_labels=True, bs=1)
        train_acc = run_inference(train_dl, ort_session)
        print(f"Train Accuracy: {train_acc:.4f}")
        
        # Validate on Validation Set
        valid_acc = run_inference(dls.valid, ort_session)
        print(f"Valid Accuracy: {valid_acc:.4f}")

except Exception as e:
    print(f"An error occurred during inference: {e}")
