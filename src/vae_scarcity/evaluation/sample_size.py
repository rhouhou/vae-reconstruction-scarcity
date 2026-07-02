"""Sample-size sweep utilities for downstream classification."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from vae_scarcity.evaluation.downstream import compute_balanced_accuracy, flatten_images
from vae_scarcity.models.classifiers import (
    build_classifier,
    fit_classifier,
    predict_labels,
)


def _to_feature_matrix(images_or_features: np.ndarray) -> np.ndarray:
    """Convert images to feature matrices if needed."""
    data = np.asarray(images_or_features)

    if data.ndim > 2:
        return flatten_images(data)

    return data


def _validate_sample_ratios(sample_ratios: Sequence[float]) -> None:
    """Validate sample-size ratios."""
    if not sample_ratios:
        raise ValueError("sample_ratios must contain at least one value.")

    for ratio in sample_ratios:
        if not 0 < ratio <= 1:
            raise ValueError(
                f"Each sample-size ratio must be in the interval (0, 1]. "
                f"Got {ratio}."
            )


def stratified_bootstrap_indices(
    labels: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample bootstrap indices while preserving class coverage.

    This function samples with replacement and ensures that each class appears
    at least once, as long as n_samples is at least the number of classes.
    """
    labels = np.asarray(labels)
    classes, class_counts = np.unique(labels, return_counts=True)
    n_classes = len(classes)

    if n_samples < n_classes:
        raise ValueError(
            f"n_samples must be at least the number of classes. "
            f"Got n_samples={n_samples}, n_classes={n_classes}."
        )

    proportions = class_counts / class_counts.sum()
    desired_counts = np.maximum(1, np.round(proportions * n_samples).astype(int))

    while desired_counts.sum() > n_samples:
        candidates = np.where(desired_counts > 1)[0]
        reduce_index = candidates[np.argmax(desired_counts[candidates])]
        desired_counts[reduce_index] -= 1

    while desired_counts.sum() < n_samples:
        add_index = np.argmax(proportions)
        desired_counts[add_index] += 1

    sampled_indices: list[np.ndarray] = []

    for class_label, count in zip(classes, desired_counts):
        class_indices = np.flatnonzero(labels == class_label)
        sampled_class_indices = rng.choice(class_indices, size=count, replace=True)
        sampled_indices.append(sampled_class_indices)

    indices = np.concatenate(sampled_indices)
    rng.shuffle(indices)

    return indices


def bootstrap_sample_size_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_ratios: Sequence[float],
    classifier_name: str = "random_forest",
    n_bootstrap: int = 20,
    random_state: int = 42,
    stratified: bool = True,
    classifier_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run a bootstrap sample-size sweep for a downstream classifier.

    Parameters
    ----------
    X_train:
        Training images or feature matrix.
    y_train:
        Training labels.
    X_test:
        Test images or feature matrix.
    y_test:
        Test labels.
    sample_ratios:
        Fractions of the training set to sample, e.g. [0.05, 0.10, 0.20].
    classifier_name:
        Classifier name supported by ``build_classifier``.
    n_bootstrap:
        Number of bootstrap repetitions per sample-size ratio.
    random_state:
        Random seed.
    stratified:
        If True, sample with class coverage.
    classifier_kwargs:
        Optional keyword arguments passed to the classifier builder.

    Returns
    -------
    pd.DataFrame
        Long-format table with one row per bootstrap run.
    """
    _validate_sample_ratios(sample_ratios)

    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be at least 1.")

    X_train = _to_feature_matrix(X_train)
    X_test = _to_feature_matrix(X_test)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same length.")

    if len(X_test) != len(y_test):
        raise ValueError("X_test and y_test must have the same length.")

    rng = np.random.default_rng(random_state)
    n_train_total = len(X_train)
    n_classes = len(np.unique(y_train))
    classifier_kwargs = classifier_kwargs or {}

    records: list[dict[str, Any]] = []

    for sample_ratio in sample_ratios:
        n_subset = int(round(n_train_total * sample_ratio))
        n_subset = max(n_subset, n_classes)
        n_subset = min(n_subset, n_train_total)

        for iteration in range(n_bootstrap):
            iteration_seed = int(rng.integers(0, np.iinfo(np.int32).max))

            if stratified:
                indices = stratified_bootstrap_indices(
                    labels=y_train,
                    n_samples=n_subset,
                    rng=rng,
                )
            else:
                indices = rng.choice(n_train_total, size=n_subset, replace=True)

            X_subset = X_train[indices]
            y_subset = y_train[indices]

            current_classifier_kwargs = dict(classifier_kwargs)
            current_classifier_kwargs.setdefault("random_state", iteration_seed)

            model = build_classifier(classifier_name, **current_classifier_kwargs)
            fitted_model = fit_classifier(model, X_subset, y_subset)
            y_pred = predict_labels(fitted_model, X_test)

            balanced_accuracy = compute_balanced_accuracy(y_test, y_pred)

            records.append(
                {
                    "classifier": classifier_name,
                    "sample_ratio": float(sample_ratio),
                    "iteration": iteration,
                    "n_train": int(n_subset),
                    "balanced_accuracy": balanced_accuracy,
                }
            )

    return pd.DataFrame.from_records(records)


def summarize_sample_size_sweep(
    results: pd.DataFrame,
    score_column: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Summarize bootstrap sample-size sweep results."""
    required_columns = {"classifier", "sample_ratio", "n_train", score_column}
    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    rows: list[dict[str, Any]] = []

    group_columns = ["classifier", "sample_ratio", "n_train"]

    for group_values, group in results.groupby(group_columns):
        classifier, sample_ratio, n_train = group_values
        scores = group[score_column].to_numpy()

        rows.append(
            {
                "classifier": classifier,
                "sample_ratio": sample_ratio,
                "n_train": n_train,
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                "n_bootstrap": int(len(scores)),
                "ci_95_low": float(np.percentile(scores, 2.5)),
                "ci_95_high": float(np.percentile(scores, 97.5)),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by=["classifier", "sample_ratio"],
        ignore_index=True,
    )