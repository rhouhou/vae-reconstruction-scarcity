import numpy as np
import pytest

from vae_scarcity.evaluation.sample_size import (
    bootstrap_sample_size_sweep,
    stratified_bootstrap_indices,
    summarize_sample_size_sweep,
)


def _toy_image_dataset():
    rng = np.random.default_rng(0)

    class_0 = rng.normal(
        loc=0.1,
        scale=0.01,
        size=(20, 8, 8, 1),
    ).astype(np.float32)

    class_1 = rng.normal(
        loc=0.9,
        scale=0.01,
        size=(20, 8, 8, 1),
    ).astype(np.float32)

    images = np.concatenate([class_0, class_1], axis=0)
    labels = np.array([0] * 20 + [1] * 20, dtype=int)

    X_train = np.concatenate([images[:15], images[20:35]], axis=0)
    y_train = np.array([0] * 15 + [1] * 15, dtype=int)

    X_test = np.concatenate([images[15:20], images[35:40]], axis=0)
    y_test = np.array([0] * 5 + [1] * 5, dtype=int)

    return X_train, y_train, X_test, y_test


def test_stratified_bootstrap_indices_preserves_class_coverage():
    rng = np.random.default_rng(42)
    labels = np.array([0] * 10 + [1] * 10)

    indices = stratified_bootstrap_indices(
        labels=labels,
        n_samples=6,
        rng=rng,
    )

    sampled_labels = labels[indices]

    assert len(indices) == 6
    assert set(sampled_labels.tolist()) == {0, 1}


def test_bootstrap_sample_size_sweep_returns_expected_rows():
    X_train, y_train, X_test, y_test = _toy_image_dataset()

    results = bootstrap_sample_size_sweep(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_ratios=[0.25, 0.50],
        classifier_name="random_forest",
        n_bootstrap=2,
        random_state=42,
        classifier_kwargs={
            "n_estimators": 10,
            "n_jobs": 1,
        },
    )

    assert len(results) == 4
    assert set(results["sample_ratio"].tolist()) == {0.25, 0.50}
    assert results["balanced_accuracy"].between(0.0, 1.0).all()


def test_summarize_sample_size_sweep_returns_one_row_per_ratio():
    X_train, y_train, X_test, y_test = _toy_image_dataset()

    results = bootstrap_sample_size_sweep(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        sample_ratios=[0.25, 0.50],
        classifier_name="random_forest",
        n_bootstrap=2,
        random_state=42,
        classifier_kwargs={
            "n_estimators": 10,
            "n_jobs": 1,
        },
    )

    summary = summarize_sample_size_sweep(results)

    assert len(summary) == 2
    assert "mean" in summary.columns
    assert "ci_95_low" in summary.columns
    assert "ci_95_high" in summary.columns


def test_invalid_sample_ratio_raises_error():
    X_train, y_train, X_test, y_test = _toy_image_dataset()

    with pytest.raises(ValueError):
        bootstrap_sample_size_sweep(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sample_ratios=[0.0],
        )


def test_invalid_bootstrap_count_raises_error():
    X_train, y_train, X_test, y_test = _toy_image_dataset()

    with pytest.raises(ValueError):
        bootstrap_sample_size_sweep(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            sample_ratios=[0.5],
            n_bootstrap=0,
        )