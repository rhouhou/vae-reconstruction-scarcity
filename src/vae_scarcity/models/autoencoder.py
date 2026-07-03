"""Autoencoder models for image reconstruction baselines."""

from __future__ import annotations

from typing import Any

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None


def _require_tensorflow() -> None:
    """Raise a helpful error if TensorFlow is not installed."""
    if tf is None:
        raise ImportError(
            "TensorFlow is required for autoencoder models. "
            'Install it with: python -m pip install -e ".[deep-learning]"'
        )


def build_denoising_autoencoder(
    image_shape: tuple[int, int, int] = (64, 64, 3),
    latent_dim: int = 256,
    noise_stddev: float = 0.10,
) -> Any:
    """Build a convolutional denoising autoencoder.

    The model adds Gaussian noise during training and learns to reconstruct
    the clean input image. During inference, the noise layer is inactive.
    """
    _require_tensorflow()

    layers = tf.keras.layers
    models = tf.keras.models

    channels = image_shape[-1]

    inputs = layers.Input(shape=image_shape, name="image_input")

    x = layers.GaussianNoise(noise_stddev, name="input_noise")(inputs)

    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(latent_dim, activation="relu", name="latent_embedding")(x)

    x = layers.Dense(4 * 4 * 256, activation="relu")(x)
    x = layers.Reshape((4, 4, 256))(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    outputs = layers.Conv2D(
        channels,
        3,
        activation="sigmoid",
        padding="same",
        name="reconstruction",
    )(x)

    return models.Model(
        inputs,
        outputs,
        name="denoising_autoencoder",
    )


def build_autoencoder_model(
    model_type: str = "denoising_autoencoder",
    image_shape: tuple[int, int, int] = (64, 64, 3),
    latent_dim: int = 256,
    noise_stddev: float = 0.10,
) -> Any:
    """Build an autoencoder model by type."""
    normalized_model_type = model_type.lower()

    if normalized_model_type in {
        "denoising_autoencoder",
        "denoising_ae",
        "dae",
    }:
        return build_denoising_autoencoder(
            image_shape=image_shape,
            latent_dim=latent_dim,
            noise_stddev=noise_stddev,
        )

    raise ValueError(
        f"Unsupported autoencoder model_type: {model_type}. "
        "Supported values are: denoising_autoencoder, denoising_ae, dae."
    )