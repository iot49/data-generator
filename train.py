#!/usr/bin/env python3

# ruff: noqa: F403, F405
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportAny=false

from pathlib import Path
import random
import matplotlib.pyplot as plt

from fastai.vision.all import *
import torch

from r49 import R49Dataset, R49DataLoaders

LABELS = ["track", "train", "other"]
MODEL_NAME = "resnet18"
MODEL_DIR = Path("models")
TUNE_STEPS = 300

R49_DIR = Path("../datasets/train-track/r49")

SIZE = 96
DB_SIZE = int(1.5*SIZE)
DPT = 30

VALID_PCT = 0.5
BATCH_SIZE = 64

# deterministic
random.seed(42)

# data
ds = R49Dataset(R49_DIR.rglob("**/*.r49"), dpt=DPT, size=int(1.5*SIZE), labels=LABELS)
dls = R49DataLoaders.from_dataset(ds, valid_pct=VALID_PCT, crop_size=SIZE, bs=BATCH_SIZE)

# train
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(TUNE_STEPS)

# save model 
# https://dev.to/tkeyo/export-fastai-resnet-models-to-onnx-2gj7
# https://www.google.com/search?q=fastai+export+onnx+model&rlz=1C9BKJA_enMX1038US1058&oq=fastai+export+onn&hl=en-US&sourceid=chrome-mobile&ie=UTF-8
# https://github.com/elliotwaite/pytorch-to-javascript-with-onnx-js/blob/master/convert_to_onnx.py
MODEL_DIR.mkdir(parents=True, exist_ok=True)
torch.save(learn.model, MODEL_DIR / f"{MODEL_NAME}.pth")

class Normalization(torch.nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))
        
    def forward(self, x):
        return (x - self.mean) / self.std

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
    torch.randn(1, 3, SIZE, SIZE), MODEL_DIR / f"{MODEL_NAME}.onnx", 
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
