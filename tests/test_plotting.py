import pandas as pd

from vae_scarcity.plotting import plot_sample_size_curve


def test_plot_sample_size_curve_creates_file(tmp_path):
    summary = pd.DataFrame(
        {
            "sample_ratio": [0.25, 0.50, 1.00],
            "mean": [0.70, 0.80, 0.90],
            "ci_95_low": [0.65, 0.75, 0.85],
            "ci_95_high": [0.75, 0.85, 0.95],
        }
    )

    output_path = tmp_path / "curve.png"

    saved_path = plot_sample_size_curve(summary, output_path)

    assert saved_path.exists()
    assert saved_path.suffix == ".png"