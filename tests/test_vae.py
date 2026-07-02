import pytest

tf = pytest.importorskip("tensorflow")

from vae_scarcity.models.vae import build_skip_vae


def test_build_skip_vae_forward_pass():
    model = build_skip_vae(
        image_shape=(64, 64, 3),
        latent_dim=16,
        kl_weight=1.0,
    )

    x = tf.random.uniform((2, 64, 64, 3))
    y = model(x)

    assert y.shape == x.shape
    assert len(model.losses) > 0