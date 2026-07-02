"""Run a toy downstream sample-size smoke experiment.

This script does not require a real dataset. It creates a tiny synthetic
two-class image dataset, runs a bootstrap sample-size sweep, and saves CSV
outputs. It is intended to verify that the downstream pipeline works end-to-end.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vae_scarcity.evaluation.sample_size import (
    bootstrap_sample_size_sweep,
    summarize_sample_size_sweep,
)


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_toy_image_dataset(
    n_train_per_class: int,
    n_test_per_class: int,
    image_size: tuple[int, int],
    channels: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a tiny synthetic two-class image dataset.

    Class 0 has low-intensity images.
    Class 1 has high-intensity images.
    """
    rng = np.random.default_rng(seed)
    height, width = image_size

    train_class_0 = rng.normal(
        loc=0.20,
        scale=0.05,
        size=(n_train_per_class, height, width, channels),
    ).astype(np.float32)

    train_class_1 = rng.normal(
        loc=0.80,
        scale=0.05,
        size=(n_train_per_class, height, width, channels),
    ).astype(np.float32)

    test_class_0 = rng.normal(
        loc=0.20,
        scale=0.05,
        size=(n_test_per_class, height, width, channels),
    ).astype(np.float32)

    test_class_1 = rng.normal(
        loc=0.80,
        scale=0.05,
        size=(n_test_per_class, height, width, channels),
    ).astype(np.float32)

    X_train = np.concatenate([train_class_0, train_class_1], axis=0)
    y_train = np.array([0] * n_train_per_class + [1] * n_train_per_class)

    X_test = np.concatenate([test_class_0, test_class_1], axis=0)
    y_test = np.array([0] * n_test_per_class + [1] * n_test_per_class)

    X_train = np.clip(X_train, 0.0, 1.0)
    X_test = np.clip(X_test, 0.0, 1.0)

    return X_train, y_train, X_test, y_test


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a toy downstream classification smoke experiment."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/downstream_smoke.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/smoke_downstream",
        help="Directory where CSV results will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the smoke experiment."""
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = int(config["seed"])
    data_config = config["data"]
    experiment_config = config["experiment"]
    classifier_config = config["classifier"]

    X_train, y_train, X_test, y_test = make_toy_image_dataset(
        n_train_per_class=int(data_config["n_train_per_class"]),
        n_test_per_class=int(data_config["n_test_per_class"]),
        image_size=tuple(data_config["image_size"]),
        channels=int(data_config["channels"]),
        seed=seed,
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
        classifier_kwargs=classifier_config,
    )

    summary = summarize_sample_size_sweep(results)

    results_path = output_dir / "downstream_smoke_results.csv"
    summary_path = output_dir / "downstream_smoke_summary.csv"

    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("Smoke downstream experiment completed.")
    print(f"Saved raw results to: {results_path}")
    print(f"Saved summary to: {summary_path}")
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()