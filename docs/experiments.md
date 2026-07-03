# Experiment Workflows

This document describes the reproducible experiment workflows in this repository.

The project studies whether VAE-reconstructed X-ray images improve downstream classification under sample scarcity.

---

## 1. Synthetic Downstream Smoke Test

This workflow does not require real data.

It creates a small synthetic two-class image dataset and runs a bootstrap sample-size sweep using a Random Forest classifier.

```bash
python scripts/run_smoke_downstream.py \
  --config configs/downstream_smoke.yaml \
  --output-dir results/smoke_downstream
```

Generate the figure:

```bash
python scripts/make_smoke_downstream_figure.py \
  --summary-csv results/smoke_downstream/downstream_smoke_summary.csv \
  --output-path results/smoke_downstream/downstream_smoke_curve.png
```

Purpose:

- test the downstream classification pipeline
- verify sample-size sweep logic
- verify CSV output
- verify plotting

Expected outputs:

```text
results/smoke_downstream/downstream_smoke_results.csv
results/smoke_downstream/downstream_smoke_summary.csv
results/smoke_downstream/downstream_smoke_curve.png
```

---

## 2. Synthetic Reconstruction Smoke Test

This workflow creates synthetic images, trains the VAE briefly, reconstructs test images, and saves reconstruction metrics.

This workflow requires TensorFlow:

```bash
python -m pip install -e ".[deep-learning]"
```

Run:

```bash
python scripts/run_smoke_reconstruction.py \
  --config configs/reconstruction_smoke.yaml \
  --output-dir results/smoke_reconstruction
```

Purpose:

- test VAE model construction
- test VAE training
- test image reconstruction
- test reconstruction metrics

Expected outputs:

```text
results/smoke_reconstruction/reconstruction_smoke_metrics.csv
results/smoke_reconstruction/reconstruction_smoke_history.csv
```

---

## 3. Original X-ray Downstream Baseline

This workflow trains a classifier directly on original X-ray images.

```bash
python scripts/run_downstream_xray_original.py \
  --config configs/downstream_xray_original.yaml \
  --data-zip data/raw/test.zip \
  --output-dir results/downstream_xray_original
```

Purpose:

- establish the raw-image baseline
- measure balanced accuracy across sample-size ratios
- save bootstrap results and summary statistics

Expected outputs:

```text
results/downstream_xray_original/original_downstream_results.csv
results/downstream_xray_original/original_downstream_summary.csv
results/downstream_xray_original/split_info.csv
```

Generate the figure:

```bash
python scripts/plot_sample_size_curve.py \
  --summary-csv results/downstream_xray_original/original_downstream_summary.csv \
  --output-path results/downstream_xray_original/original_downstream_curve.png \
  --title "Original X-ray classification under sample scarcity"
```

---

## 4. Reconstruction-Based X-ray Pipeline

This workflow trains a reconstruction model on the X-ray training split, reconstructs train and test images, and trains a classifier on the reconstructed images.

This workflow requires TensorFlow:

```bash
python -m pip install -e ".[deep-learning]"
```

Run:

```bash
python scripts/run_downstream_xray_reconstruction.py \
  --config configs/downstream_xray_skip_vae.yaml \
  --data-zip data/raw/test.zip \
  --output-dir results/downstream_xray_skip_vae
```

Purpose:

- train a VAE reconstruction model
- reconstruct train and test images
- evaluate reconstruction quality
- test downstream classification on reconstructed images

Expected outputs:

```text
results/downstream_xray_skip_vae/reconstruction_downstream_results.csv
results/downstream_xray_skip_vae/reconstruction_downstream_summary.csv
results/downstream_xray_skip_vae/reconstruction_metrics.csv
results/downstream_xray_skip_vae/reconstruction_training_history.csv
results/downstream_xray_skip_vae/split_info.csv
```

Generate the figure:

```bash
python scripts/plot_sample_size_curve.py \
  --summary-csv results/downstream_xray_skip_vae/reconstruction_downstream_summary.csv \
  --output-path results/downstream_xray_skip_vae/reconstruction_downstream_curve.png \
  --title "VAE-reconstructed X-ray classification under sample scarcity"
```

---

## 5. Original vs reconstruction Comparison

After running both X-ray workflows, compare original-image and VAE-reconstructed results.

```bash
python scripts/compare_original_vs_reconstruction.py \
  --original-results results/downstream_xray_original/original_downstream_results.csv \
  --reconstruction-results results/downstream_xray_skip_vae/reconstruction_downstream_results.csv \
  --original-summary results/downstream_xray_original/original_downstream_summary.csv \
  --reconstruction-summary results/downstream_xray_skip_vae/reconstruction_downstream_summary.csv \
  --output-dir results/comparison_original_vs_reconstruction
```

Purpose:

- combine original and VAE results
- compare mean balanced accuracy per sample-size ratio
- compute VAE minus original performance difference
- run Mann-Whitney U tests
- save a comparison curve

Expected outputs:

```text
results/comparison_original_vs_reconstruction/combined_downstream_results.csv
results/comparison_original_vs_reconstruction/combined_downstream_summary.csv
results/comparison_original_vs_reconstruction/comparison_by_sample_ratio.csv
results/comparison_original_vs_reconstruction/original_vs_reconstruction_curve.png
```

---

---

## Current Comparison Methods

The current implemented comparison includes four methods:

| Method | Description |
|---|---|
| Original images | Classifier trained directly on original X-ray images |
| Skip VAE | Classifier trained on images reconstructed with a skip-connected VAE |
| Plain VAE | Classifier trained on images reconstructed with a plain VAE without skip connections |
| Denoising AE | Classifier trained on images reconstructed with a denoising autoencoder |

The classifier is kept fixed across methods so that the comparison focuses on the reconstruction strategy.

---

## Future Extensions

Future experiments may include:

| Extension | Purpose |
|---|---|
| CNN classifier | Compare whether the reconstruction effect also appears with a deep classifier |
| More random seeds | Improve statistical robustness |
| More VAE training epochs | Test whether stronger reconstruction improves downstream performance |
| Calibration analysis | Check whether reconstructed-image classifiers are better calibrated |
| Error analysis | Identify which classes benefit or degrade after reconstruction |

These extensions are not required for the current four-method comparison.

---

## Notes

Generated outputs are stored under:

```text
results/
```

The `results/` folder is ignored by Git, except for selected legacy figures.

Do not commit generated CSV files, trained models, checkpoints, datasets, or zip files.