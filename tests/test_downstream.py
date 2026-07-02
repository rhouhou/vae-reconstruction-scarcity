import numpy as np
import pytest

from vae_scarcity.evaluation.downstream import (
    compute_balanced_accuracy,
    evaluate_classifier,
    evaluate_predictions,
    flatten_images,
)


class SimpleClassifier:
    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)


def test_flatten_images_returns_expected_shape():
    images = np.ones((10, 64, 64, 3), dtype=np.float32)

    flattened = flatten_images(images)

    assert flattened.shape == (10, 64 * 64 * 3)


def test_compute_balanced_accuracy_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    score = compute_balanced_accuracy(y_true, y_pred)

    assert score == 1.0


def test_compute_balanced_accuracy_imbalanced_all_majority_prediction():
    y_true = np.array([0] * 90 + [1] * 10)
    y_pred = np.array([0] * 100)

    score = compute_balanced_accuracy(y_true, y_pred)

    assert score == 0.5


def test_compute_balanced_accuracy_length_mismatch_raises_error():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0, 1])

    with pytest.raises(ValueError):
        compute_balanced_accuracy(y_true, y_pred)


def test_evaluate_predictions_returns_expected_keys():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])

    metrics = evaluate_predictions(y_true, y_pred)

    assert "balanced_accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert metrics["confusion_matrix"].shape == (2, 2)


def test_evaluate_classifier_with_predict_method():
    X_test = np.ones((5, 4), dtype=np.float32)
    y_test = np.zeros(5, dtype=int)
    model = SimpleClassifier()

    metrics = evaluate_classifier(model, X_test, y_test)

    assert metrics["balanced_accuracy"] == 1.0


def test_evaluate_classifier_without_predict_method_raises_error():
    X_test = np.ones((5, 4), dtype=np.float32)
    y_test = np.zeros(5, dtype=int)
    model = object()

    with pytest.raises(TypeError):
        evaluate_classifier(model, X_test, y_test)