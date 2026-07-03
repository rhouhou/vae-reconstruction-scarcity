import pytest

tf = pytest.importorskip("tensorflow")

from vae_scarcity.models.vae import build_vae_model


def test_build_vae_model_skip_vae_forward_pass():
    model = build_vae_model(
        model_type="skip_vae",
        image_shape=(64, 64, 3),
        latent_dim=8,
        kl_weight=1.0,
    )

    x = tf.random.uniform((2, 64, 64, 3))
    y = model(x)

    assert y.shape == x.shape


def test_build_vae_model_plain_vae_forward_pass():
    model = build_vae_model(
        model_type="plain_vae",
        image_shape=(64, 64, 3),
        latent_dim=8,
        kl_weight=1.0,
    )

    x = tf.random.uniform((2, 64, 64, 3))
    y = model(x)

    assert y.shape == x.shape


def test_build_vae_model_invalid_type_raises_error():
    with pytest.raises(ValueError):
        build_vae_model(
            model_type="unknown",
            image_shape=(64, 64, 3),
            latent_dim=8,
        )