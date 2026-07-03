"""Analyze sample-size efficiency from final multi-seed results.

This script uses:

results/final_analysis/all_summary_by_method_ratio.csv

and creates:

results/final_analysis/sample_size_efficiency.csv
results/final_analysis/sample_size_gaps_by_ratio.csv
results/figures/final/sample_size_efficiency_gap.png

The goal is to quantify how much labeled training data is needed to reach
near-full-data downstream performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_ratio_value(df: pd.DataFrame, ratio: float) -> float:
    """Find mean balanced accuracy for a specific sample ratio."""
    matches = df[np.isclose(df["sample_ratio"], ratio)]

    if matches.empty:
        raise ValueError(
            f"Could not find sample_ratio={ratio}. "
            f"Available ratios: {sorted(df['sample_ratio'].unique())}"
        )

    return float(matches["balanced_accuracy_mean"].iloc[0])


def minimum_ratio_for_threshold(
    method_df: pd.DataFrame,
    threshold: float,
) -> float | None:
    """Find the smallest sample ratio reaching a performance threshold."""
    ordered = method_df.sort_values("sample_ratio")

    reached = ordered[ordered["balanced_accuracy_mean"] >= threshold]

    if reached.empty:
        return None

    return float(reached["sample_ratio"].iloc[0])


def build_sample_efficiency_tables(
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build method-level and ratio-level sample-efficiency tables."""
    efficiency_rows = []
    gap_rows = []

    required_ratios = [0.20, 0.40, 0.60, 0.80, 1.00]

    for (method_key, method), method_df in summary.groupby(["method_key", "method"]):
        method_df = method_df.sort_values("sample_ratio")

        full_data_score = find_ratio_value(method_df, 1.00)

        score_20 = find_ratio_value(method_df, 0.20)
        score_40 = find_ratio_value(method_df, 0.40)
        score_60 = find_ratio_value(method_df, 0.60)
        score_80 = find_ratio_value(method_df, 0.80)

        threshold_95_percent = 0.95 * full_data_score
        threshold_within_1_point = full_data_score - 0.01
        threshold_within_2_points = full_data_score - 0.02

        min_ratio_95_percent = minimum_ratio_for_threshold(
            method_df,
            threshold_95_percent,
        )
        min_ratio_within_1_point = minimum_ratio_for_threshold(
            method_df,
            threshold_within_1_point,
        )
        min_ratio_within_2_points = minimum_ratio_for_threshold(
            method_df,
            threshold_within_2_points,
        )

        efficiency_rows.append(
            {
                "method_key": method_key,
                "method": method,
                "full_data_balanced_accuracy": full_data_score,
                "balanced_accuracy_at_20_percent": score_20,
                "balanced_accuracy_at_40_percent": score_40,
                "balanced_accuracy_at_60_percent": score_60,
                "balanced_accuracy_at_80_percent": score_80,
                "loss_20_percent_vs_full": full_data_score - score_20,
                "loss_40_percent_vs_full": full_data_score - score_40,
                "loss_60_percent_vs_full": full_data_score - score_60,
                "loss_80_percent_vs_full": full_data_score - score_80,
                "relative_performance_20_percent": score_20 / full_data_score,
                "relative_performance_40_percent": score_40 / full_data_score,
                "relative_performance_60_percent": score_60 / full_data_score,
                "relative_performance_80_percent": score_80 / full_data_score,
                "minimum_ratio_reaching_95_percent_of_full": min_ratio_95_percent,
                "minimum_ratio_within_1_point_of_full": min_ratio_within_1_point,
                "minimum_ratio_within_2_points_of_full": min_ratio_within_2_points,
            }
        )

        for ratio in required_ratios:
            score = find_ratio_value(method_df, ratio)
            gap_rows.append(
                {
                    "method_key": method_key,
                    "method": method,
                    "sample_ratio": ratio,
                    "balanced_accuracy_mean": score,
                    "full_data_balanced_accuracy": full_data_score,
                    "loss_vs_full": full_data_score - score,
                    "relative_performance_vs_full": score / full_data_score,
                }
            )

    efficiency = pd.DataFrame(efficiency_rows)
    gaps = pd.DataFrame(gap_rows)

    efficiency = efficiency.sort_values(
        "minimum_ratio_within_2_points_of_full",
        na_position="last",
    ).reset_index(drop=True)

    gaps = gaps.sort_values(["sample_ratio", "method_key"]).reset_index(drop=True)

    return efficiency, gaps


def plot_gap_by_ratio(gaps: pd.DataFrame, output_path: Path) -> None:
    """Plot performance loss versus full-data performance."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, method_df in gaps.groupby("method"):
        method_df = method_df.sort_values("sample_ratio")

        ax.plot(
            method_df["sample_ratio"],
            method_df["loss_vs_full"],
            marker="o",
            label=method,
        )

    ax.axhline(0.01, linestyle="--", linewidth=1, label="1 point gap")
    ax.axhline(0.02, linestyle=":", linewidth=1, label="2 point gap")

    ax.set_xlabel("Training sample-size ratio")
    ax.set_ylabel("Balanced accuracy loss vs full data")
    ax.set_title("Sample-size efficiency gap relative to full-data performance")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze sample-size efficiency from final results."
    )

    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("results/final_analysis/all_summary_by_method_ratio.csv"),
        help="Path to the aggregated method/ratio summary CSV.",
    )

    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("results/final_analysis"),
        help="Directory where analysis CSV files will be saved.",
    )

    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=Path("results/figures/final"),
        help="Directory where final figures will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    """Run sample-size efficiency analysis."""
    args = parse_args()

    if not args.summary_csv.exists():
        raise FileNotFoundError(
            f"Summary CSV not found: {args.summary_csv}. "
            "Run scripts/aggregate_final_results.py first."
        )

    summary = pd.read_csv(args.summary_csv)

    required_columns = {
        "method_key",
        "method",
        "sample_ratio",
        "balanced_accuracy_mean",
    }

    missing_columns = required_columns - set(summary.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(summary.columns)}"
        )

    efficiency, gaps = build_sample_efficiency_tables(summary)

    args.analysis_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    efficiency_path = args.analysis_dir / "sample_size_efficiency.csv"
    gaps_path = args.analysis_dir / "sample_size_gaps_by_ratio.csv"
    figure_path = args.figure_dir / "sample_size_efficiency_gap.png"

    efficiency.to_csv(efficiency_path, index=False)
    gaps.to_csv(gaps_path, index=False)
    plot_gap_by_ratio(gaps, figure_path)

    print("\nSaved:")
    print(efficiency_path)
    print(gaps_path)
    print(figure_path)

    print("\nSample-size efficiency:")
    print(efficiency.to_string(index=False))


if __name__ == "__main__":
    main()