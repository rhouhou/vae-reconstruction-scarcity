import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vae_scarcity.data.loaders import load_images_from_zip


def _create_test_image(path: Path, value: int) -> None:
    image = Image.fromarray(
        np.full((16, 16, 3), value, dtype=np.uint8),
        mode="RGB",
    )
    image.save(path)


def test_load_images_from_zip_with_root_dir(tmp_path):
    data_dir = tmp_path / "Dataset"
    covid_dir = data_dir / "COVID"
    normal_dir = data_dir / "NORMAL"
    covid_dir.mkdir(parents=True)
    normal_dir.mkdir(parents=True)

    _create_test_image(covid_dir / "covid_1.png", value=255)
    _create_test_image(normal_dir / "normal_1.png", value=0)

    zip_path = tmp_path / "dataset.zip"

    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for image_path in data_dir.rglob("*.png"):
            zip_file.write(image_path, image_path.relative_to(tmp_path))

    images, labels = load_images_from_zip(
        zip_path=zip_path,
        classes=["COVID", "NORMAL"],
        image_size=(64, 64),
        root_dir="Dataset",
    )

    assert images.shape == (2, 64, 64, 3)
    assert labels.tolist() == [0, 1]
    assert images.dtype == np.float32
    assert images.min() >= 0.0
    assert images.max() <= 1.0


def test_load_images_from_zip_without_root_dir(tmp_path):
    covid_dir = tmp_path / "COVID"
    normal_dir = tmp_path / "NORMAL"
    covid_dir.mkdir()
    normal_dir.mkdir()

    _create_test_image(covid_dir / "covid_1.png", value=200)
    _create_test_image(normal_dir / "normal_1.png", value=50)

    zip_path = tmp_path / "dataset.zip"

    with zipfile.ZipFile(zip_path, "w") as zip_file:
        for image_path in [covid_dir / "covid_1.png", normal_dir / "normal_1.png"]:
            zip_file.write(image_path, image_path.relative_to(tmp_path))

    images, labels = load_images_from_zip(
        zip_path=zip_path,
        classes=["COVID", "NORMAL"],
        image_size=(32, 32),
    )

    assert images.shape == (2, 32, 32, 3)
    assert labels.tolist() == [0, 1]


def test_load_images_from_zip_missing_file_raises_error(tmp_path):
    missing_zip = tmp_path / "missing.zip"

    with pytest.raises(FileNotFoundError):
        load_images_from_zip(
            zip_path=missing_zip,
            classes=["COVID", "NORMAL"],
        )


def test_load_images_from_zip_no_matching_images_raises_error(tmp_path):
    zip_path = tmp_path / "empty.zip"

    with zipfile.ZipFile(zip_path, "w") as zip_file:
        zip_file.writestr("Dataset/OTHER/file.txt", "not an image")

    with pytest.raises(ValueError):
        load_images_from_zip(
            zip_path=zip_path,
            classes=["COVID", "NORMAL"],
            root_dir="Dataset",
        )