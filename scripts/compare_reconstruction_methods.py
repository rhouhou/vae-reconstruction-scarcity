"""Compare multiple reconstruction methods in one downstream plot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find the first matching column from a list of candidates."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "Could not find any of these columns: "
        f"{candidates}. Available columns: {list(df.columns)}"
    )


def _load_method_summary(label: str, csv_path: Path) -> pd.DataFrame:
    """Load and standardize one method summary CSV."""
    df = pd.read_csv(csv_path)

    sample_ratio_col = _find_column(
        df,
        [
            "sample_ratio",
            "train_ratio",
            "ratio",
        ],
    )

    mean_col = _find_column(
        df,
        [
            "balanced_accuracy_mean",
            "mean_balanced_accuracy",
            "mean",
            "balanced_accuracy",
        ],
    )

    lower_col = None
    upper_col = None

    lower_candidates = [
        "balanced_accuracy_ci_lower",
        "ci_lower",
        "lower_ci",
        "balanced_accuracy_lower",
    ]

    upper_candidates = [
        "balanced_accuracy_ci_upper",
        "ci_upper",
        "upper_ci",
        "balanced_accuracy_upper",
    ]

    for candidate in lower_candidates:
        if candidate in df.columns:
            lower_col = candidate
            break

    for candidate in upper_candidates:
        if candidate in df.columns:
            upper_col = candidate
            break

    standardized = pd.DataFrame(
        {
            "method": label,
            "sample_ratio": df[sample_ratio_col],
            "balanced_accuracy_mean": df[mean_col],
        }
    )

    if lower_col is not None and upper_col is not None:
        standardized["balanced_accuracy_ci_lower"] = df[lower_col]
        standardized["balanced_accuracy_ci_upper"] = df[upper_col]

    return standardized


def plot_all_methods(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot all methods in one learning-curve figure."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, method_df in summary.groupby("method"):
        method_df = method_df.sort_values("sample_ratio")

        ax.plot(
            method_df["sample_ratio"],
            method_df["balanced_accuracy_mean"],
            marker="o",
            label=method,
        )

        if {
            "balanced_accuracy_ci_lower",
            "balanced_accuracy_ci_upper",
        }.issubset(method_df.columns):
            ax.fill_between(
                method_df["sample_ratio"],
                method_df["balanced_accuracy_ci_lower"],
                method_df["balanced_accuracy_ci_upper"],
                alpha=0.15,
            )

    ax.set_xlabel("Training sample-size ratio")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Downstream classification under sample scarcity")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_method_argument(value: str) -> tuple[str, Path]:
    """Parse method argument in the format Label=path/to/summary.csv."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Each --method-summary must use the format: "
            "Label=path/to/summary.csv"
        )

    label, path = value.split("=", 1)
    label = label.strip()
    path = path.strip()

    if not label:
        raise argparse.ArgumentTypeError("Method label cannot be empty.")

    if not path:
        raise argparse.ArgumentTypeError("Method CSV path cannot be empty.")

    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare downstream performance across reconstruction methods."
    )

    parser.add_argument(
        "--method-summary",
        action="append",
        required=True,
        type=parse_method_argument,
        help=(
            "Method summary in the format Label=path/to/summary.csv. "
            "Use this argument once per method."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where combined outputs will be saved.",
    )

    args = parser.parse_args()

    all_summaries = []

    for label, csv_path in args.method_summary:
        if not csv_path.exists():
            raise FileNotFoundError(f"Summary CSV not found: {csv_path}")

        all_summaries.append(_load_method_summary(label, csv_path))

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_summary = combined_summary.sort_values(["sample_ratio", "method"])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.output_dir / "all_methods_summary.csv"
    plot_path = args.output_dir / "all_methods_curve.png"

    combined_summary.to_csv(summary_path, index=False)
    plot_all_methods(combined_summary, plot_path)

    print(f"Saved combined summary to: {summary_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()