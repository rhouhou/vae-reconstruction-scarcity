"""Reconstruction quality metrics."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def mean_squared_error(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """Compute mean squared error between original and reconstructed images."""
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)
    return float(np.mean((x_true - x_pred) ** 2))


def mean_absolute_error(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """Compute mean absolute error between original and reconstructed images."""
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)
    return float(np.mean(np.abs(x_true - x_pred)))


def compute_ssim(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Compute structural similarity index for one image."""
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)

    if x_true.ndim == 3:
        return float(
            structural_similarity(
                x_true,
                x_pred,
                data_range=data_range,
                channel_axis=-1,
            )
        )

    return float(
        structural_similarity(
            x_true,
            x_pred,
            data_range=data_range,
        )
    )


def compute_psnr(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Compute peak signal-to-noise ratio for one image."""
    return float(
        peak_signal_noise_ratio(
            np.asarray(x_true),
            np.asarray(x_pred),
            data_range=data_range,
        )
    )


def compute_reconstruction_metrics(
    x_true: np.ndarray,
    x_pred: np.ndarray,
    data_range: float = 1.0,
) -> dict[str, float]:
    """Compute reconstruction metrics for one image or a batch of images.

    Parameters
    ----------
    x_true:
        Original image or image batch.
    x_pred:
        Reconstructed image or image batch.
    data_range:
        Value range of the image data. Use 1.0 for normalized images.

    Returns
    -------
    dict
        Dictionary containing MSE, MAE, SSIM, and PSNR.
    """
    x_true = np.asarray(x_true)
    x_pred = np.asarray(x_pred)

    if x_true.shape != x_pred.shape:
        raise ValueError(
            f"x_true and x_pred must have the same shape. "
            f"Got {x_true.shape} and {x_pred.shape}."
        )

    mse = mean_squared_error(x_true, x_pred)
    mae = mean_absolute_error(x_true, x_pred)

    if x_true.ndim == 4:
        ssim_values = [
            compute_ssim(x_true[i], x_pred[i], data_range=data_range)
            for i in range(x_true.shape[0])
        ]
        psnr_values = [
            compute_psnr(x_true[i], x_pred[i], data_range=data_range)
            for i in range(x_true.shape[0])
        ]
        ssim = float(np.mean(ssim_values))
        psnr = float(np.mean(psnr_values))
    else:
        ssim = compute_ssim(x_true, x_pred, data_range=data_range)
        psnr = compute_psnr(x_true, x_pred, data_range=data_range)

    return {
        "mse": mse,
        "mae": mae,
        "ssim": ssim,
        "psnr": psnr,
    }