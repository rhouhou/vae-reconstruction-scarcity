"""Compare original-image and VAE-reconstructed downstream results."""

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
        description="Compare original vs VAE-reconstructed downstream results."
    )
    parser.add_argument(
        "--original-results",
        type=str,
        default="results/downstream_xray_original/original_downstream_results.csv",
        help="Raw bootstrap results for original images.",
    )
    parser.add_argument(
        "--vae-results",
        type=str,
        default="results/downstream_xray_vae/vae_downstream_results.csv",
        help="Raw bootstrap results for VAE-reconstructed images.",
    )
    parser.add_argument(
        "--original-summary",
        type=str,
        default="results/downstream_xray_original/original_downstream_summary.csv",
        help="Summary results for original images.",
    )
    parser.add_argument(
        "--vae-summary",
        type=str,
        default="results/downstream_xray_vae/vae_downstream_summary.csv",
        help="Summary results for VAE-reconstructed images.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison_original_vs_vae",
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
    vae_results: pd.DataFrame,
    score_column: str = "balanced_accuracy",
) -> pd.DataFrame:
    """Compare original and VAE results per sample-size ratio."""
    required_columns = {"sample_ratio", score_column}

    for name, table in {
        "original_results": original_results,
        "vae_results": vae_results,
    }.items():
        missing = required_columns - set(table.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")

    rows = []

    sample_ratios = sorted(
        set(original_results["sample_ratio"]).intersection(
            set(vae_results["sample_ratio"])
        )
    )

    for ratio in sample_ratios:
        original_scores = original_results.loc[
            original_results["sample_ratio"] == ratio,
            score_column,
        ].to_numpy()

        vae_scores = vae_results.loc[
            vae_results["sample_ratio"] == ratio,
            score_column,
        ].to_numpy()

        statistic, p_value = mannwhitneyu(
            original_scores,
            vae_scores,
            alternative="two-sided",
        )

        original_mean = float(np.mean(original_scores))
        vae_mean = float(np.mean(vae_scores))
        mean_difference = vae_mean - original_mean

        rows.append(
            {
                "sample_ratio": ratio,
                "original_mean": original_mean,
                "vae_mean": vae_mean,
                "vae_minus_original": mean_difference,
                "original_std": float(np.std(original_scores, ddof=1)),
                "vae_std": float(np.std(vae_scores, ddof=1)),
                "p_value_mannwhitneyu": float(p_value),
                "n_original": int(len(original_scores)),
                "n_vae": int(len(vae_scores)),
            }
        )

    return pd.DataFrame(rows)


def plot_comparison_curve(
    original_summary: pd.DataFrame,
    vae_summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot original vs VAE-reconstructed sample-size curves."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = {"sample_ratio", "mean", "ci_95_low", "ci_95_high"}

    for name, table in {
        "original_summary": original_summary,
        "vae_summary": vae_summary,
    }.items():
        missing = required_columns - set(table.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {sorted(missing)}")

    original_summary = original_summary.sort_values("sample_ratio")
    vae_summary = vae_summary.sort_values("sample_ratio")

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
        vae_summary["sample_ratio"],
        vae_summary["mean"],
        marker="o",
        label="VAE-reconstructed images",
    )
    plt.fill_between(
        vae_summary["sample_ratio"],
        vae_summary["ci_95_low"],
        vae_summary["ci_95_high"],
        alpha=0.20,
    )

    plt.xlabel("Training sample-size ratio")
    plt.ylabel("Balanced accuracy")
    plt.title("Original vs VAE-reconstructed X-ray classification")
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
    vae_results = _load_csv(args.vae_results)
    original_summary = _load_csv(args.original_summary)
    vae_summary = _load_csv(args.vae_summary)

    original_results = original_results.copy()
    vae_results = vae_results.copy()
    original_summary = original_summary.copy()
    vae_summary = vae_summary.copy()

    original_results["pipeline"] = "original"
    vae_results["pipeline"] = "vae_reconstructed"
    original_summary["pipeline"] = "original"
    vae_summary["pipeline"] = "vae_reconstructed"

    combined_results = pd.concat(
        [original_results, vae_results],
        ignore_index=True,
    )
    combined_summary = pd.concat(
        [original_summary, vae_summary],
        ignore_index=True,
    )

    comparison = compare_by_sample_ratio(
        original_results=original_results,
        vae_results=vae_results,
    )

    combined_results_path = output_dir / "combined_downstream_results.csv"
    combined_summary_path = output_dir / "combined_downstream_summary.csv"
    comparison_path = output_dir / "comparison_by_sample_ratio.csv"
    figure_path = output_dir / "original_vs_vae_curve.png"

    combined_results.to_csv(combined_results_path, index=False)
    combined_summary.to_csv(combined_summary_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    saved_figure = plot_comparison_curve(
        original_summary=original_summary,
        vae_summary=vae_summary,
        output_path=figure_path,
    )

    print("Original vs VAE comparison completed.")
    print(f"Saved combined raw results to: {combined_results_path}")
    print(f"Saved combined summary to: {combined_summary_path}")
    print(f"Saved comparison table to: {comparison_path}")
    print(f"Saved figure to: {saved_figure}")
    print()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()