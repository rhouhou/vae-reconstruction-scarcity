"""Generic reconstruction model builder."""

from __future__ import annotations

from typing import Any

from vae_scarcity.models.autoencoder import build_autoencoder_model
from vae_scarcity.models.vae import build_vae_model


def build_reconstruction_model(
    model_type: str = "skip_vae",
    image_shape: tuple[int, int, int] = (64, 64, 3),
    latent_dim: int = 256,
    kl_weight: float = 1.0,
    noise_stddev: float = 0.10,
) -> Any:
    """Build a reconstruction model by type.

    Supported model types:
    - skip_vae
    - plain_vae
    - denoising_autoencoder
    - denoising_ae
    - dae
    """
    normalized_model_type = model_type.lower()

    if normalized_model_type in {"skip_vae", "skip", "plain_vae", "plain"}:
        return build_vae_model(
            model_type=normalized_model_type,
            image_shape=image_shape,
            latent_dim=latent_dim,
            kl_weight=kl_weight,
        )

    if normalized_model_type in {
        "denoising_autoencoder",
        "denoising_ae",
        "dae",
    }:
        return build_autoencoder_model(
            model_type=normalized_model_type,
            image_shape=image_shape,
            latent_dim=latent_dim,
            noise_stddev=noise_stddev,
        )

    raise ValueError(
        f"Unsupported reconstruction model_type: {model_type}. "
        "Supported values are: skip_vae, plain_vae, denoising_autoencoder."
    )