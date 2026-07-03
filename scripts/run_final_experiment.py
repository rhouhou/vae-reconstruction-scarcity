"""Run the final multi-method experiment for one random seed.

This script reads configs/final_experiment.yaml and runs all final methods
for a single seed:

- Original images
- Skip VAE reconstructions
- Plain VAE reconstructions
- Denoising autoencoder reconstructions
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file."""
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    return data


def write_yaml(data: dict[str, Any], path: Path) -> None:
    """Write a YAML file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)


def build_runtime_config(
    base_config_path: Path,
    final_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Create a seed-specific runtime config from a method config."""
    runtime_config = load_yaml(base_config_path)

    runtime_config["seed"] = seed

    final_data = final_config.get("data", {})
    runtime_config.setdefault("data", {})

    for key in ["image_size", "classes", "root_dir"]:
        if key in final_data:
            runtime_config["data"][key] = final_data[key]

    final_experiment = final_config.get("experiment", {})
    runtime_config.setdefault("experiment", {})

    for key in ["sample_ratios", "n_bootstrap", "classifier"]:
        if key in final_experiment:
            runtime_config["experiment"][key] = final_experiment[key]

    # Normalize old config naming if any method config still uses "vae".
    if "vae" in runtime_config and "reconstruction" not in runtime_config:
        runtime_config["reconstruction"] = runtime_config.pop("vae")

    return runtime_config


def run_command(command: list[str], dry_run: bool = False) -> None:
    """Run a command or print it in dry-run mode."""
    printable = " ".join(command)
    print(f"\nRunning:\n{printable}\n")

    if dry_run:
        return

    subprocess.run(command, cwd=REPO_ROOT, check=True)


def run_method(
    method_key: str,
    method_config: dict[str, Any],
    final_config: dict[str, Any],
    seed: int,
    seed_output_dir: Path,
    data_zip: Path,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> None:
    """Run one method for one seed."""
    kind = method_config["kind"]
    base_config_path = REPO_ROOT / method_config["config"]
    output_dir = seed_output_dir / method_config["output_subdir"]
    summary_path = output_dir / method_config["summary_file"]

    if skip_existing and summary_path.exists():
        print(f"Skipping existing method output: {method_key} -> {summary_path}")
        return

    runtime_config = build_runtime_config(
        base_config_path=base_config_path,
        final_config=final_config,
        seed=seed,
    )

    generated_config_path = (
        seed_output_dir / "generated_configs" / f"{method_key}.yaml"
    )
    write_yaml(runtime_config, generated_config_path)

    if kind == "original":
        script_path = REPO_ROOT / "scripts" / "run_downstream_xray_original.py"
    elif kind == "reconstruction":
        script_path = (
            REPO_ROOT / "scripts" / "run_downstream_xray_reconstruction.py"
        )
    else:
        raise ValueError(
            f"Unsupported method kind for {method_key}: {kind}. "
            "Supported values are: original, reconstruction."
        )

    command = [
        sys.executable,
        str(script_path),
        "--config",
        str(generated_config_path),
        "--data-zip",
        str(data_zip),
        "--output-dir",
        str(output_dir),
    ]

    run_command(command, dry_run=dry_run)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the final four-method experiment for one seed."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/final_experiment.yaml"),
        help="Path to the final experiment protocol config.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed to run.",
    )

    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help=(
            "Optional subset of method keys to run. "
            "Example: --methods original skip_vae"
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a method if its summary CSV already exists.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the final experiment for one seed."""
    args = parse_args()

    final_config_path = REPO_ROOT / args.config
    final_config = load_yaml(final_config_path)

    allowed_seeds = final_config.get("seeds", [])

    if args.seed not in allowed_seeds:
        raise ValueError(
            f"Seed {args.seed} is not listed in {final_config_path}. "
            f"Allowed seeds are: {allowed_seeds}"
        )

    data_zip = REPO_ROOT / final_config["data"]["data_zip"]
    output_root = REPO_ROOT / final_config["outputs"]["output_root"]
    seed_output_dir = output_root / f"seed_{args.seed}"

    methods = final_config["methods"]

    selected_method_keys = args.methods if args.methods is not None else list(methods)

    unknown_methods = sorted(set(selected_method_keys) - set(methods))
    if unknown_methods:
        raise ValueError(
            f"Unknown methods: {unknown_methods}. "
            f"Available methods are: {list(methods)}"
        )

    print(f"Final experiment config: {final_config_path}")
    print(f"Seed: {args.seed}")
    print(f"Data zip: {data_zip}")
    print(f"Output directory: {seed_output_dir}")
    print(f"Methods: {selected_method_keys}")

    for method_key in selected_method_keys:
        print(f"\n===== Method: {method_key} =====")

        run_method(
            method_key=method_key,
            method_config=methods[method_key],
            final_config=final_config,
            seed=args.seed,
            seed_output_dir=seed_output_dir,
            data_zip=data_zip,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
        )

    print("\nFinished final experiment run.")


if __name__ == "__main__":
    main()