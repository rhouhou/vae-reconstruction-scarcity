"""Data loading utilities for image datasets."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def load_images_from_zip(
    zip_path: str | Path,
    classes: Sequence[str],
    image_size: tuple[int, int] = (64, 64),
    root_dir: str | None = None,
    normalize: bool = True,
    color_mode: str = "rgb",
) -> tuple[np.ndarray, np.ndarray]:
    """Load labeled images from a zip archive.

    Expected zip structure:

        root_dir/class_name/image.png

    or, if root_dir is None:

        class_name/image.png

    Parameters
    ----------
    zip_path:
        Path to the zip file containing image folders.
    classes:
        Class folder names. The label is assigned by the class order.
    image_size:
        Target image size as (width, height).
    root_dir:
        Optional root directory inside the zip file.
    normalize:
        If True, scale image values to [0, 1].
    color_mode:
        Either "rgb" or "grayscale".

    Returns
    -------
    images:
        Array of images with shape (n_images, height, width, channels).
    labels:
        Integer labels with shape (n_images,).
    """
    zip_path = Path(zip_path)

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    if color_mode not in {"rgb", "grayscale"}:
        raise ValueError("color_mode must be either 'rgb' or 'grayscale'.")

    images: list[np.ndarray] = []
    labels: list[int] = []

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        names = zip_ref.namelist()

        for label, class_name in enumerate(classes):
            if root_dir:
                prefix = f"{root_dir.rstrip('/')}/{class_name}/"
            else:
                prefix = f"{class_name}/"

            class_files = [
                name
                for name in names
                if name.startswith(prefix)
                and name.lower().endswith(IMAGE_EXTENSIONS)
                and not name.endswith("/")
            ]

            for image_name in sorted(class_files):
                with zip_ref.open(image_name) as image_file:
                    image = Image.open(image_file)

                    if color_mode == "rgb":
                        image = image.convert("RGB")
                    else:
                        image = image.convert("L")

                    image = image.resize(image_size)
                    image_array = np.asarray(image)

                    if color_mode == "grayscale":
                        image_array = image_array[..., np.newaxis]

                    if normalize:
                        image_array = image_array.astype(np.float32) / 255.0
                    else:
                        image_array = image_array.astype(np.uint8)

                    images.append(image_array)
                    labels.append(label)

    if not images:
        raise ValueError(
            "No images found. Check zip_path, root_dir, class names, and file structure."
        )

    return np.stack(images), np.asarray(labels, dtype=np.int64)