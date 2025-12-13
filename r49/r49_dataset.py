from pathlib import Path
from typing import Callable

import torch
from cv2.typing import MatLike
from PIL import Image

from .image_transform import apply_perspective_transform
from .manifest import Manifest
from .r49_file import R49File


class R49Dataset(torch.utils.data.ConcatDataset[tuple[Image.Image, str]]):
    def __init__(
        self,
        r49_files: list[Path],
        *,
        dpt: int = 20,
        size: int = 64,
        labels: list[str] | None = None,
        image_transform: Callable[
            [MatLike, Manifest, int], tuple[MatLike, MatLike]
        ] = apply_perspective_transform,
    ):
        labels = labels if labels is not None else ["track", "train", "other"]
        super().__init__( 
            [
                R49File(r49_file, image_transform=image_transform, size=size, labels=labels)
                for r49_file in r49_files
            ]
        )
