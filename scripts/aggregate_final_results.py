"""Aggregate final multi-seed reconstruction experiment results.

This script combines the final five-seed experiment outputs into paper-ready
summary tables and figures.

Expected input structure:

results/final_runs/
├── seed_42/
├── seed_123/
├── seed_777/
├── seed_2024/
└── seed_2025/

Expected outputs:

results/final_analysis/all_results.csv
results/final_analysis/seed_method_ratio_summary.csv
results/final_analysis/all_summary_by_method_ratio.csv
results/final_analysis/sample_efficiency_auc_by_seed.csv
results/final_analysis/sample_efficiency_auc.csv
results/final_analysis/split_summary.csv
results/final_analysis/final_multiseed_curve.png
results/figures/final/final_multiseed_curve.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML config."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    return data


def find_column(df: pd.DataFrame, candidates: list[str]) -> str:
    """Find the first matching column from a list of possible names."""
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        f"Could not find any of columns {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def load_method_results(
    path: Path,
    seed: int,
    method_key: str,
    method_label: str,
) -> pd.DataFrame:
    """Load raw downstream results for one method and one seed."""
    if not path.exists():
        raise FileNotFoundError(f"Missing results file: {path}")

    df = pd.read_csv(path)

    sample_ratio_col = find_column(
        df,
        ["sample_ratio", "train_ratio", "ratio"],
    )

    metric_col = find_column(
        df,
        [
            "balanced_accuracy",
            "balanced_accuracy_mean",
            "mean_balanced_accuracy",
            "score",
            "metric",
        ],
    )

    standardized = pd.DataFrame(
        {
            "seed": seed,
            "method_key": method_key,
            "method": method_label,
            "sample_ratio": df[sample_ratio_col].astype(float),
            "balanced_accuracy": df[metric_col].astype(float),
        }
    )

    if "bootstrap_idx" in df.columns:
        standardized["bootstrap_idx"] = df["bootstrap_idx"]
    elif "bootstrap_iteration" in df.columns:
        standardized["bootstrap_idx"] = df["bootstrap_iteration"]
    else:
        standardized["bootstrap_idx"] = np.arange(len(df))

    return standardized


def load_split_info(path: Path, seed: int, method_key: str, method_label: str) -> pd.DataFrame:
    """Load split_info.csv if available."""
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.insert(0, "method", method_label)
    df.insert(0, "method_key", method_key)
    df.insert(0, "seed", seed)

    return df


def summarize_by_seed_method_ratio(all_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize bootstrap results within each seed/method/sample-ratio."""
    summary = (
        all_results.groupby(["seed", "method_key", "method", "sample_ratio"])
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            balanced_accuracy_min=("balanced_accuracy", "min"),
            balanced_accuracy_max=("balanced_accuracy", "max"),
            n_bootstrap=("balanced_accuracy", "count"),
        )
        .reset_index()
    )

    summary["balanced_accuracy_std"] = summary["balanced_accuracy_std"].fillna(0.0)

    return summary


def summarize_across_seeds(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """Summarize seed-level means across seeds.

    The uncertainty reported here is across random seeds, not just bootstrap
    repeats. This is the preferred summary for final interpretation.
    """
    rows = []

    grouped = seed_summary.groupby(["method_key", "method", "sample_ratio"])

    for (method_key, method, sample_ratio), group in grouped:
        values = group["balanced_accuracy_mean"].to_numpy(dtype=float)
        n_seeds = len(values)

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n_seeds > 1 else 0.0
        sem = std / np.sqrt(n_seeds) if n_seeds > 1 else 0.0

        if n_seeds > 1:
            t_value = stats.t.ppf(0.975, df=n_seeds - 1)
            ci_half_width = float(t_value * sem)
        else:
            ci_half_width = 0.0

        rows.append(
            {
                "method_key": method_key,
                "method": method,
                "sample_ratio": sample_ratio,
                "balanced_accuracy_mean": mean,
                "balanced_accuracy_std_across_seeds": std,
                "balanced_accuracy_sem_across_seeds": sem,
                "balanced_accuracy_ci_lower": mean - ci_half_width,
                "balanced_accuracy_ci_upper": mean + ci_half_width,
                "n_seeds": n_seeds,
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["sample_ratio", "method_key"]).reset_index(drop=True)

    return summary


def compute_auc_by_seed(seed_summary: pd.DataFrame) -> pd.DataFrame:
    """Compute area under the sample-size curve for each method and seed."""
    rows = []

    grouped = seed_summary.groupby(["seed", "method_key", "method"])

    for (seed, method_key, method), group in grouped:
        group = group.sort_values("sample_ratio")

        x = group["sample_ratio"].to_numpy(dtype=float)
        y = group["balanced_accuracy_mean"].to_numpy(dtype=float)

        auc = float(np.trapz(y, x))

        rows.append(
            {
                "seed": seed,
                "method_key": method_key,
                "method": method,
                "sample_efficiency_auc": auc,
            }
        )

    return pd.DataFrame(rows)


def summarize_auc(auc_by_seed: pd.DataFrame) -> pd.DataFrame:
    """Summarize sample-efficiency AUC across seeds."""
    rows = []

    grouped = auc_by_seed.groupby(["method_key", "method"])

    for (method_key, method), group in grouped:
        values = group["sample_efficiency_auc"].to_numpy(dtype=float)
        n_seeds = len(values)

        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if n_seeds > 1 else 0.0
        sem = std / np.sqrt(n_seeds) if n_seeds > 1 else 0.0

        if n_seeds > 1:
            t_value = stats.t.ppf(0.975, df=n_seeds - 1)
            ci_half_width = float(t_value * sem)
        else:
            ci_half_width = 0.0

        rows.append(
            {
                "method_key": method_key,
                "method": method,
                "sample_efficiency_auc_mean": mean,
                "sample_efficiency_auc_std": std,
                "sample_efficiency_auc_sem": sem,
                "sample_efficiency_auc_ci_lower": mean - ci_half_width,
                "sample_efficiency_auc_ci_upper": mean + ci_half_width,
                "n_seeds": n_seeds,
            }
        )

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("sample_efficiency_auc_mean", ascending=False)

    return summary


def add_difference_vs_original(summary: pd.DataFrame) -> pd.DataFrame:
    """Add method minus original difference for each sample ratio."""
    original = summary[summary["method_key"] == "original"][
        ["sample_ratio", "balanced_accuracy_mean"]
    ].rename(columns={"balanced_accuracy_mean": "original_balanced_accuracy_mean"})

    merged = summary.merge(original, on="sample_ratio", how="left")

    merged["difference_vs_original"] = (
        merged["balanced_accuracy_mean"]
        - merged["original_balanced_accuracy_mean"]
    )

    return merged


def plot_multiseed_curve(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot mean balanced accuracy with across-seed 95% CI."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, method_df in summary.groupby("method"):
        method_df = method_df.sort_values("sample_ratio")

        ax.plot(
            method_df["sample_ratio"],
            method_df["balanced_accuracy_mean"],
            marker="o",
            label=method,
        )

        ax.fill_between(
            method_df["sample_ratio"],
            method_df["balanced_accuracy_ci_lower"],
            method_df["balanced_accuracy_ci_upper"],
            alpha=0.15,
        )

    ax.set_xlabel("Training sample-size ratio")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Final multi-seed downstream comparison")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Aggregate final multi-seed experiment results."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final_experiment.yaml"),
        help="Path to final experiment config.",
    )

    return parser.parse_args()


def main() -> None:
    """Aggregate final experiment results."""
    args = parse_args()

    final_config = load_yaml(args.config)

    seeds = final_config["seeds"]
    methods = final_config["methods"]

    output_root = Path(final_config["outputs"]["output_root"])
    analysis_dir = Path(final_config["outputs"]["analysis_dir"])
    figure_dir = Path(final_config["outputs"]["figure_dir"])

    all_results = []
    all_split_info = []

    for seed in seeds:
        seed_root = output_root / f"seed_{seed}"

        for method_key, method_config in methods.items():
            method_label = method_config["label"]
            method_dir = seed_root / method_config["output_subdir"]

            results_path = method_dir / method_config["results_file"]
            split_info_path = method_dir / "split_info.csv"

            print(f"Loading seed={seed}, method={method_key}: {results_path}")

            method_results = load_method_results(
                path=results_path,
                seed=seed,
                method_key=method_key,
                method_label=method_label,
            )

            all_results.append(method_results)

            split_info = load_split_info(
                path=split_info_path,
                seed=seed,
                method_key=method_key,
                method_label=method_label,
            )

            if not split_info.empty:
                all_split_info.append(split_info)

    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df = all_results_df.sort_values(
        ["seed", "method_key", "sample_ratio", "bootstrap_idx"]
    )

    seed_summary = summarize_by_seed_method_ratio(all_results_df)
    final_summary = summarize_across_seeds(seed_summary)
    final_summary = add_difference_vs_original(final_summary)

    auc_by_seed = compute_auc_by_seed(seed_summary)
    auc_summary = summarize_auc(auc_by_seed)

    all_results_path = analysis_dir / "all_results.csv"
    seed_summary_path = analysis_dir / "seed_method_ratio_summary.csv"
    final_summary_path = analysis_dir / "all_summary_by_method_ratio.csv"
    auc_by_seed_path = analysis_dir / "sample_efficiency_auc_by_seed.csv"
    auc_summary_path = analysis_dir / "sample_efficiency_auc.csv"
    analysis_plot_path = analysis_dir / "final_multiseed_curve.png"
    final_figure_path = figure_dir / "final_multiseed_curve.png"

    all_results_df.to_csv(all_results_path, index=False)
    seed_summary.to_csv(seed_summary_path, index=False)
    final_summary.to_csv(final_summary_path, index=False)
    auc_by_seed.to_csv(auc_by_seed_path, index=False)
    auc_summary.to_csv(auc_summary_path, index=False)

    if all_split_info:
        split_summary = pd.concat(all_split_info, ignore_index=True)
        split_summary.to_csv(analysis_dir / "split_summary.csv", index=False)

    plot_multiseed_curve(final_summary, analysis_plot_path)
    plot_multiseed_curve(final_summary, final_figure_path)

    print("\nSaved:")
    print(all_results_path)
    print(seed_summary_path)
    print(final_summary_path)
    print(auc_by_seed_path)
    print(auc_summary_path)
    print(analysis_plot_path)
    print(final_figure_path)


if __name__ == "__main__":
    main()