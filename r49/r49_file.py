
import json
import zipfile
from pathlib import Path
from typing import Callable

from numpy.typing import NDArray

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
from cv2.typing import MatLike
from PIL import Image

from .image_transform import apply_perspective_transform
from .manifest import Manifest


class R49File(torch.utils.data.Dataset[tuple[Image.Image, str]]):  
    def __init__(
        self,
        r49file: Path,
        *,
        dpt: int = 20,
        size: int = 64,
        image_transform: Callable[
            [MatLike, Manifest, int], tuple[MatLike, MatLike]
        ] = apply_perspective_transform,  # type: ignore
        rotation_angles: list[float] | None = None,
        verbose: bool = False,
    ):
        self._r49file: Path = r49file
        self._size: int = size
        self._image: MatLike | None = None  # cv2 image (_read_r49)
        self._image_transform: Callable[[MatLike, Manifest, int], tuple[MatLike, NDArray[np.float32]]] = image_transform
        self._dpt: int = dpt
        self._rotation_angles: list[float] = rotation_angles if rotation_angles is not None else [0]
        self._verbose: bool = verbose
        self._manifest: Manifest
        self._x: list[MatLike] = []
        self._y: list[str] = []

        self._read_r49()
        self._create_xy()
        
        if self._image is None or self._image.size == 0:
            # This might happen if no images were processed in _create_xy
            raise ValueError(f"Failed to load or process any images from {self._r49file}")

    @property
    def manifest(self):
        return self._manifest

    @property
    def image(self):
        return self._image

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx: int):
        img_mat = self._x[idx]
        # Match FastAI's ToTensor: BGR->RGB, HWC->CHW, 0-255->0-1 float
        # But return PIL Image, so FastAI can handle the rest
        img_cv2_rgb: MatLike = cv2.cvtColor(img_mat, cv2.COLOR_BGR2RGB)  
        pil_img = Image.fromarray(img_cv2_rgb) 
        return pil_img, self._y[idx]        

    def save(self, output_path: Path):
        """Save the dataset samples to the specified output path."""
        for i, (img, label) in enumerate(zip(self._x, self._y)):
            label_dir = output_path / label
            label_dir.mkdir(parents=True, exist_ok=True)
            im_file = label_dir / f"{label}.{self._r49file.stem}_{i}.jpg"
            cv2.imwrite(str(im_file), img)

    def _read_r49(self):
        with zipfile.ZipFile(self._r49file, "r") as zf:
            with zf.open("manifest.json") as manifest_file:
                manifest_dict = json.load(manifest_file) 
                self._manifest = Manifest(**manifest_dict)
                assert self._manifest.version == 2, (
                    f"Got manifest unsupported version {self._manifest.version}. Expected version 2."
                )

    def _create_xy(self):
        size = self._size

        with zipfile.ZipFile(self._r49file, "r") as zf:
            for i in range(self._manifest.number_of_images):
                image_meta = self._manifest.get_image(i)
                filename = image_meta.filename
                
                # Read image bytes from zip
                try:
                    with zf.open(filename) as img_file:
                        image_bytes = img_file.read()
                except KeyError:
                    # Try finding the file if exact match fails (e.g. ./ prefix issues)
                    # or just raise
                    raise ValueError(f"Image file {filename} not found in {self._r49file}")

                # Convert bytes to numpy array and decode with OpenCV
                nparr = np.frombuffer(image_bytes, np.uint8)
                image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                # Apply transform
                # Note: valid only if calibration is global and applies to all images, 
                # OR if transform uses camera/calibration from manifest which is global.
                # In V2, calibration is global (Manifest.calibration).
                transformed_image, transform_matrix = self._image_transform(
                    image=image_cv2,
                    manifest=self._manifest,
                    dpt=self._dpt,
                )
                
                # Keep reference to last processed image for property
                self._image = transformed_image 

                for label_id, marker in image_meta.labels.items():
                    # Use marker type as label
                    label_name = marker.type
                    
                    if label_name in ["train-end", "coupling", "other"]:
                        # continue
                        pass

                    lx, ly = marker.x, marker.y

                    for angle in self._rotation_angles:
                        # Transform the marker position to the transformed image coordinate space
                        marker_point = np.array([[[lx, ly]]], dtype=np.float32)
                        [[[cx_float, cy_float]]] = cv2.perspectiveTransform(
                            marker_point, transform_matrix
                        )

                        cx = int(cx_float)
                        cy = int(cy_float)

                        # Calculate region size needed for rotation
                        region_size = int(size * 1.5)
                        radius = region_size // 2

                        try:
                            # Check bounds
                            _ = transformed_image[cy - radius, cx - radius]
                            _ = transformed_image[cy + radius - 1, cx + radius - 1]

                            # Rotate
                            rotated_region = cv2.warpAffine(
                                transformed_image[
                                    cy - radius : cy + radius, cx - radius : cx + radius
                                ],
                                cv2.getRotationMatrix2D((radius, radius), angle, 1.0),
                                (region_size, region_size),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0),
                            )

                            cropped_image = rotated_region[
                                radius - size // 2 : radius + size // 2,
                                radius - size // 2 : radius + size // 2,
                            ]

                        except (IndexError, cv2.error):
                            if self._verbose:
                                print(
                                    f"Skipping {label_id} in {filename}: out of bounds after transform."
                                )
                            break

                        self._x.append(cropped_image)
                        self._y.append(label_name)

    def __str__(self):
        return f"R49FileDataset(Path('{self._r49file}'))"
