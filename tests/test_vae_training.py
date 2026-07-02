import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from vae_scarcity.models.vae import build_skip_vae
from vae_scarcity.training.vae_training import (
    compile_vae,
    reconstruct_images,
    train_vae,
)


def test_compile_vae():
    model = build_skip_vae(
        image_shape=(64, 64, 3),
        latent_dim=8,
    )

    compiled_model = compile_vae(model, learning_rate=1e-4)

    assert compiled_model.optimizer is not None


def test_reconstruct_images_shape():
    model = build_skip_vae(
        image_shape=(64, 64, 3),
        latent_dim=8,
    )

    compile_vae(model)

    images = np.random.rand(2, 64, 64, 3).astype(np.float32)
    reconstructions = reconstruct_images(model, images, batch_size=1)

    assert reconstructions.shape == images.shape


def test_train_vae_one_epoch():
    model = build_skip_vae(
        image_shape=(64, 64, 3),
        latent_dim=8,
    )

    X_train = np.random.rand(4, 64, 64, 3).astype(np.float32)
    X_val = np.random.rand(2, 64, 64, 3).astype(np.float32)

    history = train_vae(
        model=model,
        X_train=X_train,
        X_val=X_val,
        batch_size=2,
        epochs=1,
        learning_rate=1e-4,
    )

    assert "loss" in history.history