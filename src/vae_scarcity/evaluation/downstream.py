"""Downstream classification evaluation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix


def flatten_images(images: np.ndarray) -> np.ndarray:
    """Flatten image arrays for classical machine learning classifiers.

    Parameters
    ----------
    images:
        Image array with shape (n_samples, height, width, channels).

    Returns
    -------
    np.ndarray
        Flattened image array with shape (n_samples, n_features).
    """
    images = np.asarray(images)

    if images.ndim < 2:
        raise ValueError("images must have at least 2 dimensions.")

    return images.reshape(images.shape[0], -1)


def compute_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Compute balanced accuracy from true and predicted labels."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred must have the same length. "
            f"Got {y_true.shape[0]} and {y_pred.shape[0]}."
        )

    return float(balanced_accuracy_score(y_true, y_pred))


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    """Evaluate classification predictions.

    Parameters
    ----------
    y_true:
        Ground-truth integer labels.
    y_pred:
        Predicted integer labels.

    Returns
    -------
    dict
        Dictionary with balanced accuracy and confusion matrix.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "balanced_accuracy": compute_balanced_accuracy(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def evaluate_classifier(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, Any]:
    """Evaluate a fitted classifier using balanced accuracy.

    The classifier must implement a ``predict`` method.
    """
    if not hasattr(model, "predict"):
        raise TypeError("model must implement a predict method.")

    y_pred = model.predict(X_test)

    return evaluate_predictions(y_test, y_pred)