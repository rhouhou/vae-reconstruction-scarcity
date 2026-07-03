"""Run downstream sample-size sweep on original X-ray images.

This script loads a real image dataset from a zip file, creates a reproducible
train/validation/test split, runs a bootstrap sample-size sweep on original
images, and saves CSV outputs.

It does not use VAE reconstructions yet. This is the baseline pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from vae_scarcity.data.loaders import load_images_from_zip
from vae_scarcity.data.splits import train_val_test_split
from vae_scarcity.evaluation.sample_size import (
    bootstrap_sample_size_sweep,
    summarize_sample_size_sweep,
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run original-image downstream X-ray sample-size sweep."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/downstream_xray_original.yaml",
        help="Path to downstream X-ray YAML config.",
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
        default="results/downstream_xray_original",
        help="Directory where outputs will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run downstream X-ray baseline experiment."""
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    data_config = config["data"]
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

    results = bootstrap_sample_size_sweep(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
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

    results_path = output_dir / "original_downstream_results.csv"
    summary_path = output_dir / "original_downstream_summary.csv"
    split_path = output_dir / "split_info.csv"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    pd.DataFrame([split_info]).to_csv(split_path, index=False)

    print("Original-image downstream experiment completed.")
    print(f"Saved raw results to: {results_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved split info to: {split_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()