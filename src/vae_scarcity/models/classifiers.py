"""Classifier model builders."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


ClassifierName = Literal["random_forest", "rf", "svm", "svc"]


def build_random_forest(
    n_estimators: int = 100,
    random_state: int = 42,
    class_weight: str | dict | None = "balanced",
    n_jobs: int | None = -1,
    **kwargs: Any,
) -> RandomForestClassifier:
    """Build a Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=n_jobs,
        **kwargs,
    )


def build_svm(
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: str | float = "scale",
    class_weight: str | dict | None = "balanced",
    probability: bool = False,
    random_state: int = 42,
    **kwargs: Any,
) -> SVC:
    """Build a Support Vector Machine classifier."""
    return SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        probability=probability,
        random_state=random_state,
        **kwargs,
    )


def build_classifier(
    name: ClassifierName,
    **kwargs: Any,
) -> RandomForestClassifier | SVC:
    """Build a classifier by name.

    Supported names:
    - "random_forest" or "rf"
    - "svm" or "svc"
    """
    normalized_name = name.lower()

    if normalized_name in {"random_forest", "rf"}:
        return build_random_forest(**kwargs)

    if normalized_name in {"svm", "svc"}:
        return build_svm(**kwargs)

    raise ValueError(
        f"Unsupported classifier: {name}. "
        "Supported classifiers are: random_forest, rf, svm, svc."
    )


def fit_classifier(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Any:
    """Fit a classifier and return the fitted model."""
    if not hasattr(model, "fit"):
        raise TypeError("model must implement a fit method.")

    model.fit(X_train, y_train)
    return model


def predict_labels(
    model: Any,
    X: np.ndarray,
) -> np.ndarray:
    """Predict class labels from a fitted classifier."""
    if not hasattr(model, "predict"):
        raise TypeError("model must implement a predict method.")

    return np.asarray(model.predict(X))