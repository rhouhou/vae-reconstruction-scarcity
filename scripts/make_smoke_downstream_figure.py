"""Make a figure from downstream smoke experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vae_scarcity.plotting import plot_sample_size_curve


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a sample-size curve from smoke experiment results."
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        default="results/smoke_downstream/downstream_smoke_summary.csv",
        help="Path to downstream smoke summary CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="results/smoke_downstream/downstream_smoke_curve.png",
        help="Path where the figure will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    """Create the figure."""
    args = parse_args()

    summary_path = Path(args.summary_csv)

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Summary CSV not found: {summary_path}. "
            "Run scripts/run_smoke_downstream.py first."
        )

    summary = pd.read_csv(summary_path)

    output_path = plot_sample_size_curve(
        summary=summary,
        output_path=args.output_path,
        title="Smoke test: downstream classification under sample scarcity",
    )

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()