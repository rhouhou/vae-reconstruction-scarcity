"""Plotting utilities for VAE scarcity experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_sample_size_curve(
    summary: pd.DataFrame,
    output_path: str | Path,
    x_column: str = "sample_ratio",
    y_column: str = "mean",
    low_column: str = "ci_95_low",
    high_column: str = "ci_95_high",
    title: str = "Downstream classification under sample scarcity",
    ylabel: str = "Balanced accuracy",
) -> Path:
    """Plot balanced accuracy across sample-size ratios.

    Parameters
    ----------
    summary:
        Summary table produced by summarize_sample_size_sweep.
    output_path:
        Path where the figure will be saved.
    x_column:
        Column containing sample-size ratios.
    y_column:
        Column containing mean score.
    low_column:
        Column containing lower confidence interval.
    high_column:
        Column containing upper confidence interval.
    title:
        Figure title.
    ylabel:
        Y-axis label.

    Returns
    -------
    Path
        Saved figure path.
    """
    required_columns = {x_column, y_column, low_column, high_column}
    missing_columns = required_columns - set(summary.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = summary.sort_values(x_column)

    x = summary[x_column].to_numpy()
    y = summary[y_column].to_numpy()
    y_low = summary[low_column].to_numpy()
    y_high = summary[high_column].to_numpy()

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o", label="Mean balanced accuracy")
    plt.fill_between(x, y_low, y_high, alpha=0.25, label="95% CI")

    plt.xlabel("Training sample-size ratio")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path