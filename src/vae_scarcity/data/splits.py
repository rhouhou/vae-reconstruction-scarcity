"""Dataset splitting utilities."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def train_val_test_split(
    images: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    stratify: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split images and labels into train, validation, and test sets.

    Parameters
    ----------
    images:
        Image array with shape (n_samples, height, width, channels).
    labels:
        Label array with shape (n_samples,).
    test_size:
        Fraction of total data used for the test set.
    val_size:
        Fraction of total data used for the validation set.
    random_state:
        Random seed for reproducibility.
    stratify:
        If True, preserve class proportions in each split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    images = np.asarray(images)
    labels = np.asarray(labels)

    if len(images) != len(labels):
        raise ValueError(
            f"images and labels must have the same length. "
            f"Got {len(images)} and {len(labels)}."
        )

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not 0 < val_size < 1:
        raise ValueError("val_size must be between 0 and 1.")

    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be less than 1.")

    stratify_labels = labels if stratify else None

    X_temp, X_test, y_temp, y_test = train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    relative_val_size = val_size / (1.0 - test_size)
    stratify_temp = y_temp if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=stratify_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test