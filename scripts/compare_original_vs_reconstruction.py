"""Compare original-image and Reconstruction-reconstructed downstream results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare original vs Reconstruction-reconstructed downstream results."
    )
    parser.add_argument(
        "--original-results",
        type=str,
        default="results/downstream_xray_original/original_downstream_results.csv",
        help="Raw bootstrap results for original images.",
    )
    parser.add_argument(
        "--reconstruction-results",
        type=str,
        default="results/downstream_xray_skip_reconstruction/reconstruction_downstream_results.csv",
        help="Raw bootstrap results for Reconstruction-reconstructed images.",
    )
    parser.add_argument(
        "--original-summary",
        type=str,
        default="results/downstream_xray_original/original_downstream_summary.csv",
        help="Summary results for original images.",
    )
    parser.add_argument(
        "--reconstruction-summary",
        type=str,
        default="results/downstream_xray_skip_reconstruction/reconstruction_downstream_summary.csv",
        help="Summary results for Reconstruction-reconstructed images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison_original_vs_reconstruction",
        help="Directory where comparison outputs will be saved.",
    )
    return parser.parse_args()


def _load_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV file with a clear error if missing."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return pd.read_csv(path)


def compare_by_sample_ratio(
    original_results: pd.DataFrame,
    reconstruction_results: pd.DataFrame,
    score_column: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Compare original and Reconstruction results per sample-size ratio."""
    required_columns = {"sample_ratio", score_column}

    for name, table in {
        "original_results": original_results,
        "reconstruction_results": reconstruction_results,
    }.items():
        missing = required_columns - set(table.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")

    rows = []

    sample_ratios = sorted(
        set(original_results["sample_ratio"]).intersection(
            set(reconstruction_results["sample_ratio"])
        )
    )

    for ratio in sample_ratios:
        original_scores = original_results.loc[
            original_results["sample_ratio"] == ratio,
            score_column,
        ].to_numpy()

        reconstruction_scores = reconstruction_results.loc[
            reconstruction_results["sample_ratio"] == ratio,
            score_column,
        ].to_numpy()

        statistic, p_value = mannwhitneyu(
            original_scores,
            reconstruction_scores,
            alternative="two-sided",
        )

        original_mean = float(np.mean(original_scores))
        reconstruction_mean = float(np.mean(reconstruction_scores))
        mean_difference = reconstruction_mean - original_mean

        rows.append(
            {
                "sample_ratio": ratio,
                "original_mean": original_mean,
                "reconstruction_mean": reconstruction_mean,
                "reconstruction_minus_original": mean_difference,
                "original_std": float(np.std(original_scores, ddof=1)),
                "reconstruction_std": float(np.std(reconstruction_scores, ddof=1)),
                "p_value_mannwhitneyu": float(p_value),
                "n_original": int(len(original_scores)),
                "n_reconstruction": int(len(reconstruction_scores)),
            }
        )

    return pd.DataFrame(rows)


def plot_comparison_curve(
    original_summary: pd.DataFrame,
    reconstruction_summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot original vs Reconstruction-reconstructed sample-size curves."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {"sample_ratio", "mean", "ci_95_low", "ci_95_high"}

    for name, table in {
        "original_summary": original_summary,
        "reconstruction_summary": reconstruction_summary,
    }.items():
        missing = required_columns - set(table.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")

    original_summary = original_summary.sort_values("sample_ratio")
    reconstruction_summary = reconstruction_summary.sort_values("sample_ratio")

    plt.figure(figsize=(8, 5))

    plt.plot(
        original_summary["sample_ratio"],
        original_summary["mean"],
        marker="o",
        label="Original images",
    )
    plt.fill_between(
        original_summary["sample_ratio"],
        original_summary["ci_95_low"],
        original_summary["ci_95_high"],
        alpha=0.20,
    )

    plt.plot(
        reconstruction_summary["sample_ratio"],
        reconstruction_summary["mean"],
        marker="o",
        label="Reconstruction-reconstructed images",
    )
    plt.fill_between(
        reconstruction_summary["sample_ratio"],
        reconstruction_summary["ci_95_low"],
        reconstruction_summary["ci_95_high"],
        alpha=0.20,
    )

    plt.xlabel("Training sample-size ratio")
    plt.ylabel("Balanced accuracy")
    plt.title("Original vs Reconstruction-reconstructed X-ray classification")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def main() -> None:
    """Run comparison."""
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_results = _load_csv(args.original_results)
    reconstruction_results = _load_csv(args.reconstruction_results)
    original_summary = _load_csv(args.original_summary)
    reconstruction_summary = _load_csv(args.reconstruction_summary)

    original_results = original_results.copy()
    reconstruction_results = reconstruction_results.copy()
    original_summary = original_summary.copy()
    reconstruction_summary = reconstruction_summary.copy()

    original_results["pipeline"] = "original"
    reconstruction_results["pipeline"] = "reconstruction_reconstructed"
    original_summary["pipeline"] = "original"
    reconstruction_summary["pipeline"] = "reconstruction_reconstructed"

    combined_results = pd.concat(
        [original_results, reconstruction_results],
        ignore_index=True,
    )
    combined_summary = pd.concat(
        [original_summary, reconstruction_summary],
        ignore_index=True,
    )

    comparison = compare_by_sample_ratio(
        original_results=original_results,
        reconstruction_results=reconstruction_results,
    )

    combined_results_path = output_dir / "combined_downstream_results.csv"
    combined_summary_path = output_dir / "combined_downstream_summary.csv"
    comparison_path = output_dir / "comparison_by_sample_ratio.csv"
    figure_path = output_dir / "original_vs_reconstruction_curve.png"

    combined_results.to_csv(combined_results_path, index=False)
    combined_summary.to_csv(combined_summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    saved_figure = plot_comparison_curve(
        original_summary=original_summary,
        reconstruction_summary=reconstruction_summary,
        output_path=figure_path,
    )

    print("Original vs Reconstruction comparison completed.")
    print(f"Saved combined raw results to: {combined_results_path}")
    print(f"Saved combined summary to: {combined_summary_path}")
    print(f"Saved comparison table to: {comparison_path}")
    print(f"Saved figure to: {saved_figure}")
    print()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()