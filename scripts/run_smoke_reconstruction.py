"""Run a toy VAE reconstruction smoke experiment.

This script creates a tiny synthetic image dataset, trains the skip-connected
VAE for a very small number of epochs, reconstructs test images, and saves
reconstruction metrics.

It is intended as a lightweight end-to-end check, not as a scientific result.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from vae_scarcity.evaluation.reconstruction import compute_reconstruction_metrics
from vae_scarcity.models.vae import build_skip_vae
from vae_scarcity.training.vae_training import reconstruct_images, train_vae


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_toy_reconstruction_dataset(
    n_samples: int,
    image_size: tuple[int, int],
    channels: int,
    seed: int,
) -> np.ndarray:
    """Create simple synthetic images for reconstruction testing.

    The images contain smooth gradients plus small random noise.
    """
    rng = np.random.default_rng(seed)
    height, width = image_size

    x_axis = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_axis = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x_axis, y_axis)

    base = 0.5 * xx + 0.5 * yy
    base = base[..., np.newaxis]

    if channels > 1:
        base = np.repeat(base, channels, axis=-1)

    images = []

    for _ in range(n_samples):
        noise = rng.normal(
            loc=0.0,
            scale=0.05,
            size=(height, width, channels),
        ).astype(np.float32)

        image = np.clip(base + noise, 0.0, 1.0)
        images.append(image)

    return np.stack(images).astype(np.float32)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a toy VAE reconstruction smoke experiment."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/reconstruction_smoke.yaml",
        help="Path to reconstruction smoke YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/smoke_reconstruction",
        help="Directory where outputs will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run reconstruction smoke experiment."""
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    data_config = config["data"]
    vae_config = config["vae"]

    image_size = tuple(data_config["image_size"])
    channels = int(data_config["channels"])
    image_shape = (image_size[0], image_size[1], channels)

    X_train = make_toy_reconstruction_dataset(
        n_samples=int(data_config["n_train"]),
        image_size=image_size,
        channels=channels,
        seed=seed,
    )

    X_val = make_toy_reconstruction_dataset(
        n_samples=int(data_config["n_val"]),
        image_size=image_size,
        channels=channels,
        seed=seed + 1,
    )

    X_test = make_toy_reconstruction_dataset(
        n_samples=int(data_config["n_test"]),
        image_size=image_size,
        channels=channels,
        seed=seed + 2,
    )

    model = build_skip_vae(
        image_shape=image_shape,
        latent_dim=int(vae_config["latent_dim"]),
        kl_weight=float(vae_config["kl_weight"]),
    )

    history = train_vae(
        model=model,
        X_train=X_train,
        X_val=X_val,
        batch_size=int(vae_config["batch_size"]),
        epochs=int(vae_config["epochs"]),
        learning_rate=float(vae_config["learning_rate"]),
    )

    X_reconstructed = reconstruct_images(
        model=model,
        images=X_test,
        batch_size=int(vae_config["batch_size"]),
    )

    metrics = compute_reconstruction_metrics(
        X_test,
        X_reconstructed,
        data_range=1.0,
    )

    metrics_path = output_dir / "reconstruction_smoke_metrics.csv"
    history_path = output_dir / "reconstruction_smoke_history.csv"

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    pd.DataFrame(history.history).to_csv(history_path, index=False)

    print("Smoke reconstruction experiment completed.")
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved training history to: {history_path}")
    print()
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()