"""Training utilities for VAE reconstruction models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None


def _require_tensorflow() -> None:
    """Raise a clear error if TensorFlow is not installed."""
    if tf is None:
        raise ImportError(
            "TensorFlow is required for VAE training. "
            'Install it with: python -m pip install -e ".[deep-learning]"'
        )


def compile_vae(
    model: Any,
    learning_rate: float = 1e-4,
    loss: str = "mse",
) -> Any:
    """Compile a VAE model for image reconstruction."""
    _require_tensorflow()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss)

    return model


def train_vae(
    model: Any,
    X_train: np.ndarray,
    X_val: np.ndarray | None = None,
    batch_size: int = 64,
    epochs: int = 50,
    learning_rate: float = 1e-4,
    patience: int = 10,
) -> Any:
    """Train a VAE on image reconstruction.

    The model is trained to reconstruct its own input:

        X -> VAE -> X_hat
    """
    _require_tensorflow()

    compile_vae(model, learning_rate=learning_rate)

    callbacks: list[Any] = []

    if X_val is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        )

    validation_data = (X_val, X_val) if X_val is not None else None

    history = model.fit(
        X_train,
        X_train,
        validation_data=validation_data,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    return history


def reconstruct_images(
    model: Any,
    images: np.ndarray,
    batch_size: int = 64,
) -> np.ndarray:
    """Reconstruct images using a trained VAE."""
    _require_tensorflow()

    reconstructions = model.predict(
        images,
        batch_size=batch_size,
        verbose=0,
    )

    return np.asarray(reconstructions)


def save_vae_model(
    model: Any,
    output_path: str | Path,
) -> Path:
    """Save a trained VAE model."""
    _require_tensorflow()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(output_path)

    return output_path