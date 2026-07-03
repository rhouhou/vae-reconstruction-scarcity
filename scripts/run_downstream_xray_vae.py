"""Run downstream sample-size sweep on VAE-reconstructed X-ray images.

This script loads a real image dataset from a zip file, trains a VAE on the
training split, reconstructs train/test images, evaluates reconstruction
quality, and runs a downstream sample-size sweep on the reconstructed images.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from vae_scarcity.data.loaders import load_images_from_zip
from vae_scarcity.data.splits import train_val_test_split
from vae_scarcity.evaluation.reconstruction import compute_reconstruction_metrics
from vae_scarcity.evaluation.sample_size import (
    bootstrap_sample_size_sweep,
    summarize_sample_size_sweep,
)
from vae_scarcity.models.vae import build_skip_vae
from vae_scarcity.training.vae_training import reconstruct_images, train_vae


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run VAE-reconstructed X-ray downstream sample-size sweep."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/downstream_xray_vae.yaml",
        help="Path to downstream X-ray VAE YAML config.",
    )
    parser.add_argument(
        "--data-zip",
        type=str,
        required=True,
        help="Path to the X-ray dataset zip file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/downstream_xray_vae",
        help="Directory where outputs will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run VAE-reconstructed downstream X-ray experiment."""
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    data_config = config["data"]
    vae_config = config["vae"]
    experiment_config = config["experiment"]
    classifier_config = config["classifier"]

    images, labels = load_images_from_zip(
        zip_path=args.data_zip,
        classes=data_config["classes"],
        image_size=tuple(data_config["image_size"]),
        root_dir=data_config.get("root_dir"),
        normalize=bool(data_config.get("normalize", True)),
        color_mode=data_config.get("color_mode", "rgb"),
    )

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        images=images,
        labels=labels,
        test_size=float(data_config["test_size"]),
        val_size=float(data_config["val_size"]),
        random_state=seed,
        stratify=True,
    )

    image_size = tuple(data_config["image_size"])
    channels = X_train.shape[-1]
    image_shape = (image_size[0], image_size[1], channels)

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
        patience=int(vae_config["patience"]),
    )

    X_train_vae = reconstruct_images(
        model=model,
        images=X_train,
        batch_size=int(vae_config["batch_size"]),
    )

    X_test_vae = reconstruct_images(
        model=model,
        images=X_test,
        batch_size=int(vae_config["batch_size"]),
    )

    reconstruction_metrics = compute_reconstruction_metrics(
        X_test,
        X_test_vae,
        data_range=1.0,
    )

    results = bootstrap_sample_size_sweep(
        X_train=X_train_vae,
        y_train=y_train,
        X_test=X_test_vae,
        y_test=y_test,
        sample_ratios=experiment_config["sample_ratios"],
        classifier_name=experiment_config["classifier"],
        n_bootstrap=int(experiment_config["n_bootstrap"]),
        random_state=seed,
        stratified=True,
        classifier_kwargs=classifier_config,
    )

    summary = summarize_sample_size_sweep(results)

    split_info = {
        "n_total": len(images),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "classes": ",".join(data_config["classes"]),
    }

    results_path = output_dir / "vae_downstream_results.csv"
    summary_path = output_dir / "vae_downstream_summary.csv"
    reconstruction_metrics_path = output_dir / "vae_reconstruction_metrics.csv"
    history_path = output_dir / "vae_training_history.csv"
    split_path = output_dir / "split_info.csv"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame([reconstruction_metrics]).to_csv(
        reconstruction_metrics_path,
        index=False,
    )
    pd.DataFrame(history.history).to_csv(history_path, index=False)
    pd.DataFrame([split_info]).to_csv(split_path, index=False)

    print("VAE-reconstructed downstream experiment completed.")
    print(f"Saved raw downstream results to: {results_path}")
    print(f"Saved downstream summary to: {summary_path}")
    print(f"Saved reconstruction metrics to: {reconstruction_metrics_path}")
    print(f"Saved VAE training history to: {history_path}")
    print(f"Saved split info to: {split_path}")
    print()
    print("Reconstruction metrics:")
    for key, value in reconstruction_metrics.items():
        print(f"{key}: {value:.6f}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()