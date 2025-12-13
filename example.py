#!/usr/bin/env python3

# ruff: noqa: F403, F405
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false
# pyright: reportAny=false

import shutil
from pathlib import Path

from fastai.vision.all import *

from r49 import R49File, R49Dataset, R49DataLoaders

SIZE = 64
DB_SIZE = int(1.5*SIZE)
DPT = 20

R49_DIR = Path("../datasets/train-track/r49")

# deterministic
random.seed(42)

ds = R49Dataset(R49_DIR.rglob("**/*.r49"), dpt=DPT, size=int(1.5*SIZE))
dls = R49DataLoaders.from_dataset(ds, valid_pct=0.5, crop_size=SIZE, bs=64)

# get first image of first batch, display shape, label and image
b = dls.train.one_batch()

print(f"Batch shape: x={b[0].shape}, y={b[1].shape}")
print(f"First item: x={b[0][0].shape}, y={b[1][0]}")

# Show some samples from the training set
dls.train.show_batch(max_n=12)
plt.show()