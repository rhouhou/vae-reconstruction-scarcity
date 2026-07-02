import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from vae_scarcity.models.classifiers import (
    build_classifier,
    build_random_forest,
    build_svm,
    fit_classifier,
    predict_labels,
)


def _toy_classification_data():
    X_train = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [1.0, 1.0],
            [1.1, 1.0],
            [1.0, 1.1],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    return X_train, y_train


def test_build_random_forest_returns_model():
    model = build_random_forest(n_estimators=10, random_state=42)

    assert isinstance(model, RandomForestClassifier)


def test_build_svm_returns_model():
    model = build_svm()

    assert isinstance(model, SVC)


def test_build_classifier_random_forest_aliases():
    model_1 = build_classifier("random_forest", n_estimators=10)
    model_2 = build_classifier("rf", n_estimators=10)

    assert isinstance(model_1, RandomForestClassifier)
    assert isinstance(model_2, RandomForestClassifier)


def test_build_classifier_svm_aliases():
    model_1 = build_classifier("svm")
    model_2 = build_classifier("svc")

    assert isinstance(model_1, SVC)
    assert isinstance(model_2, SVC)


def test_build_classifier_invalid_name_raises_error():
    with pytest.raises(ValueError):
        build_classifier("unknown")


def test_fit_classifier_and_predict_labels_random_forest():
    X_train, y_train = _toy_classification_data()

    model = build_random_forest(n_estimators=20, random_state=42)
    fitted_model = fit_classifier(model, X_train, y_train)
    predictions = predict_labels(fitted_model, X_train)

    assert predictions.shape == y_train.shape
    assert set(predictions.tolist()).issubset({0, 1})


def test_fit_classifier_without_fit_method_raises_error():
    X_train, y_train = _toy_classification_data()

    with pytest.raises(TypeError):
        fit_classifier(object(), X_train, y_train)


def test_predict_labels_without_predict_method_raises_error():
    X_train, _ = _toy_classification_data()

    with pytest.raises(TypeError):
        predict_labels(object(), X_train)