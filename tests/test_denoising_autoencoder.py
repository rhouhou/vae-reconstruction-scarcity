import pytest

tf = pytest.importorskip("tensorflow")

from vae_scarcity.models.autoencoder import (
    build_autoencoder_model,
    build_denoising_autoencoder,
)


def test_build_denoising_autoencoder_forward_pass():
    model = build_denoising_autoencoder(
        image_shape=(64, 64, 3),
        latent_dim=16,
        noise_stddev=0.10,
    )

    x = tf.random.uniform((2, 64, 64, 3))
    y = model(x, training=False)

    assert y.shape == x.shape


def test_build_autoencoder_model_denoising_forward_pass():
    model = build_autoencoder_model(
        model_type="denoising_autoencoder",
        image_shape=(64, 64, 3),
        latent_dim=16,
        noise_stddev=0.10,
    )

    x = tf.random.uniform((2, 64, 64, 3))
    y = model(x, training=False)

    assert y.shape == x.shape


def test_build_autoencoder_model_invalid_type_raises_error():
    with pytest.raises(ValueError):
        build_autoencoder_model(
            model_type="unknown",
            image_shape=(64, 64, 3),
            latent_dim=16,
        )