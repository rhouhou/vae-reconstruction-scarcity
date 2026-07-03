"""Plot a sample-size curve from a downstream summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from vae_scarcity.plotting import plot_sample_size_curve


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a sample-size curve from a downstream summary CSV."
    )
    parser.add_argument(
        "--summary-csv",
        type=str,
        required=True,
        help="Path to the downstream summary CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path where the figure will be saved.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Downstream classification under sample scarcity",
        help="Figure title.",
    )
    return parser.parse_args()


def main() -> None:
    """Create the sample-size curve figure."""
    args = parse_args()

    summary_path = Path(args.summary_csv)

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")

    summary = pd.read_csv(summary_path)

    output_path = plot_sample_size_curve(
        summary=summary,
        output_path=args.output_path,
        title=args.title,
    )

    print(f"Saved figure to: {output_path}")


if __name__ == "__main__":
    main()