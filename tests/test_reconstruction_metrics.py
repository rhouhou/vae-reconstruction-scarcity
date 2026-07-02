import numpy as np
import pytest

from vae_scarcity.evaluation.reconstruction import (
    compute_reconstruction_metrics,
    mean_absolute_error,
    mean_squared_error,
)


def test_mse_is_zero_for_identical_images():
    image = np.ones((64, 64, 3), dtype=np.float32)

    mse = mean_squared_error(image, image)

    assert mse == 0.0


def test_mae_is_zero_for_identical_images():
    image = np.ones((64, 64, 3), dtype=np.float32)

    mae = mean_absolute_error(image, image)

    assert mae == 0.0


def test_reconstruction_metrics_identical_batch():
    images = np.ones((2, 64, 64, 3), dtype=np.float32)

    metrics = compute_reconstruction_metrics(images, images)

    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["ssim"] == pytest.approx(1.0)
    assert np.isinf(metrics["psnr"])


def test_reconstruction_metrics_shape_mismatch_raises_error():
    x_true = np.ones((64, 64, 3), dtype=np.float32)
    x_pred = np.ones((32, 32, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        compute_reconstruction_metrics(x_true, x_pred)