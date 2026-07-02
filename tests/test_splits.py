import numpy as np
import pytest

from vae_scarcity.data.splits import train_val_test_split


def test_train_val_test_split_shapes():
    images = np.ones((100, 64, 64, 3), dtype=np.float32)
    labels = np.array([0] * 50 + [1] * 50)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        images,
        labels,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
    )

    assert X_train.shape[0] == 60
    assert X_val.shape[0] == 20
    assert X_test.shape[0] == 20

    assert y_train.shape[0] == 60
    assert y_val.shape[0] == 20
    assert y_test.shape[0] == 20


def test_train_val_test_split_is_reproducible():
    images = np.arange(100 * 4).reshape(100, 2, 2, 1)
    labels = np.array([0] * 50 + [1] * 50)

    split_1 = train_val_test_split(
        images,
        labels,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
    )

    split_2 = train_val_test_split(
        images,
        labels,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
    )

    for array_1, array_2 in zip(split_1, split_2):
        np.testing.assert_array_equal(array_1, array_2)


def test_train_val_test_split_length_mismatch_raises_error():
    images = np.ones((10, 64, 64, 3), dtype=np.float32)
    labels = np.ones((9,), dtype=np.int64)

    with pytest.raises(ValueError):
        train_val_test_split(images, labels)


def test_train_val_test_split_invalid_sizes_raise_error():
    images = np.ones((10, 64, 64, 3), dtype=np.float32)
    labels = np.array([0] * 5 + [1] * 5)

    with pytest.raises(ValueError):
        train_val_test_split(images, labels, test_size=0.6, val_size=0.5)