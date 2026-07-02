# VAE Reconstruction Under Sample Scarcity

This repository studies how variational autoencoder (VAE) reconstruction behaves when training data are limited, and how VAE-reconstructed biomedical images affect downstream classification under sample scarcity.

The project combines two related research directions:

1. **Reconstruction under sample scarcity**  
   Evaluating how well a VAE can reconstruct images when only limited training data are available.

2. **Downstream classification under sample scarcity**  
   Comparing classifiers trained on original images versus VAE-reconstructed images across different training sample sizes.

The goal is to build a clean, reproducible, and portfolio-ready research project around VAE reconstruction, uncertainty, learning curves, and biomedical image classification.

---

## Motivation

Biomedical imaging tasks often suffer from limited labeled data. In low-data regimes, machine learning models can become unstable, overfit, or require more labeled examples than are practically available.

This project explores whether VAE-based reconstruction can act as a learned denoising, compression, or regularization step before downstream classification.

The central research question is:

> Can VAE-reconstructed biomedical images improve reconstruction stability and downstream classification performance when labeled training data are scarce?

---

## Project Overview

The repository is organized around two experiment tracks.

### Track 1: VAE Reconstruction

This track evaluates the VAE as an image reconstruction model.

The basic pipeline is:

```text
input image -> VAE encoder -> latent representation -> VAE decoder -> reconstructed image
```

Evaluation metrics include:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Structural Similarity Index (SSIM)
- Peak Signal-to-Noise Ratio (PSNR)
- Reconstruction stability under perturbations
- Optional uncertainty estimation

---

### Track 2: Classification on Original vs Reconstructed Images

This track compares downstream classification performance using:

```text
original images -> classifier -> balanced accuracy
```

and:

```text
original images -> trained VAE -> reconstructed images -> classifier -> balanced accuracy
```

The experiment varies the amount of available training data using sample-size ratios such as:

```text
0.05, 0.10, 0.15, ..., 0.80
```

For each sample-size ratio, classifiers are trained and evaluated repeatedly using bootstrap resampling.

Supported or planned classifiers include:

- Random Forest
- Convolutional Neural Network
- Support Vector Machine

The main downstream metric is:

- Balanced accuracy

Additional analyses include:

- 95% confidence intervals
- Statistical comparison between original and reconstructed pipelines
- Learning-curve fitting
- Sample-efficiency analysis

---

## Preliminary Legacy Results

This project builds on an earlier exploratory VAE experiment using biomedical X-ray images.

In the earlier version, a VAE was trained to reconstruct X-ray images, and downstream classifiers were compared on:

- original X-ray images
- VAE-reconstructed X-ray images

Preliminary results suggested that VAE-reconstructed images may improve classifier stability in some low-sample regimes.

These results should be treated as exploratory until they are reproduced using the cleaned and reproducible experiment pipeline in this repository.

Example legacy figures may include:

```text
results/figures/legacy/balance_accuracy_testing_checker.png
results/figures/legacy/balance_accuracy_testing_skin.png
results/figures/legacy/box_plot_both.png
results/figures/legacy/cnn_learning_curve.png
results/figures/legacy/cnn_samplesize_pvalues.png
results/figures/legacy/cnn_vae_learning_curve.png
```

The legacy experiments motivated the current project structure. Final claims should be based only on reproducible runs from the cleaned pipeline.

---

## Repository Structure

Planned cleaned structure:

```text
vae-reconstruction-scarcity/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── reconstruction_smoke.yaml
│   ├── downstream_xray_smoke.yaml
│   └── downstream_xray_full.yaml
├── src/
│   └── vae_scarcity/
│       ├── __init__.py
│       ├── data/
│       │   ├── loaders.py
│       │   └── splits.py
│       ├── models/
│       │   ├── vae.py
│       │   └── classifiers.py
│       ├── evaluation/
│       │   ├── reconstruction.py
│       │   ├── downstream.py
│       │   ├── statistics.py
│       │   └── learning_curves.py
│       ├── experiments/
│       │   ├── run_reconstruction_sweep.py
│       │   └── run_downstream_sweep.py
│       └── plotting.py
├── scripts/
│   ├── run_smoke_reconstruction.py
│   ├── run_smoke_downstream.py
│   └── make_figures.py
├── results/
│   ├── figures/
│   └── metrics/
├── legacy/
│   └── old_xray_vae/
└── tests/
    ├── test_vae_forward.py
    ├── test_metrics.py
    └── test_downstream_smoke.py
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/rhouhou/vae-reconstruction-scarcity.git
cd vae-reconstruction-scarcity
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

---

## Quick Start

### Reconstruction smoke test

```bash
python scripts/run_smoke_reconstruction.py \
  --config configs/reconstruction_smoke.yaml \
  --output-dir results/smoke_reconstruction
```

### Downstream classification smoke test

```bash
python scripts/run_smoke_downstream.py \
  --config configs/downstream_xray_smoke.yaml \
  --data-zip data/raw/xray_dataset.zip \
  --output-dir results/smoke_downstream
```

The smoke tests are intended to verify that the pipeline runs end-to-end. They are not intended to produce final scientific results.

---

## Full Downstream Experiment

A full sample-scarcity sweep can be run with:

```bash
python -m vae_scarcity.experiments.run_downstream_sweep \
  --config configs/downstream_xray_full.yaml \
  --data-zip data/raw/xray_dataset.zip \
  --output-dir results/downstream_xray_full
```

The full experiment is expected to:

1. Load and preprocess the image dataset.
2. Train or load a VAE.
3. Reconstruct training and test images.
4. Train classifiers on original images.
5. Train classifiers on reconstructed images.
6. Evaluate balanced accuracy across sample-size ratios.
7. Save metrics, confidence intervals, statistical tests, and figures.

---

## Example Configuration

Example downstream experiment configuration:

```yaml
seed: 42

data:
  image_size: [64, 64]
  classes:
    - COVID19
    - NORMAL
    - PNEUMONIA

vae:
  model_type: skip_vae
  latent_dim: 256
  batch_size: 64
  epochs: 100
  checkpoint_path: models/vae_xray

experiment:
  sample_ratios: [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
  n_bootstrap: 20
  classifiers:
    - random_forest
    - cnn

metrics:
  reconstruction:
    - mse
    - mae
    - ssim
    - psnr
  downstream:
    - balanced_accuracy

output:
  save_predictions: false
  save_reconstructions: true
  save_figures: true
```

---

## Metrics

### Reconstruction Metrics

| Metric | Description |
|---|---|
| MSE | Mean squared pixel-wise reconstruction error |
| MAE | Mean absolute pixel-wise reconstruction error |
| SSIM | Structural similarity between original and reconstructed image |
| PSNR | Peak signal-to-noise ratio |

### Classification Metrics

| Metric | Description |
|---|---|
| Balanced accuracy | Average recall across classes |
| Confidence interval | Bootstrap-based uncertainty around performance |
| Learning curve | Performance as a function of training sample size |
| Statistical test | Comparison between original and reconstructed pipelines |

---

## Learning-Curve Analysis

The downstream classification experiments can be summarized using learning curves.

A typical inverse power-law model is:

```text
error(n) = a * n^(-b) + c
```

where:

- `n` is the training sample size or sample-size ratio
- `a` controls the initial error scale
- `b` controls the learning rate
- `c` estimates the irreducible error or performance floor

This allows comparison between the original-image pipeline and the VAE-reconstructed-image pipeline.

---

## Legacy Code

The folder:

```text
legacy/old_xray_vae/
```

contains earlier exploratory scripts from the original VAE/X-ray project.

These files are kept for transparency and historical reference. They are not the recommended entry point for running experiments.

The cleaned and reusable implementation should live under:

```text
src/vae_scarcity/
```

---

## Current Status

This repository is under active cleanup and restructuring.

Planned improvements include:

- Refactor legacy scripts into reusable modules.
- Replace hard-coded local and Colab paths with configuration files.
- Add reproducible smoke tests.
- Add CI with GitHub Actions.
- Add cleaned reconstruction and downstream experiment scripts.
- Add full result tables and final figures.
- Add experiment cards for completed runs.
- Add clear documentation for datasets, checkpoints, and limitations.

---

## Limitations

This project is currently intended for research, education, and portfolio demonstration.

Important limitations:

- Datasets are not included in the repository.
- Trained model checkpoints are not included.
- Large generated results are not included.
- Legacy figures are preliminary and should not be interpreted as final scientific evidence.
- Performance may depend strongly on dataset quality, class imbalance, preprocessing, VAE architecture, and classifier choice.
- The VAE may remove diagnostically relevant details if reconstruction quality is insufficient.
- Reconstructed images should not be assumed to be clinically faithful without careful validation.

---

## Reproducibility Notes

To make experiments reproducible, cleaned runs should record:

- random seed
- dataset version
- train/validation/test split
- image preprocessing steps
- VAE architecture
- latent dimension
- training epochs
- optimizer settings
- classifier type
- sample-size ratios
- number of bootstrap iterations
- evaluation metrics
- software versions

---

## Disclaimer

This repository is for research, education, and portfolio demonstration only.

It is not intended for clinical diagnosis, treatment planning, medical decision-making, or deployment in healthcare settings.

Biomedical image reconstruction and classification require careful validation before any real-world use.

---

## License

This project is released under the MIT License.

See the `LICENSE` file for details.

---

## Suggested Citation

If you use or build on this repository, please cite it as:

```text
Houhou, R. VAE Reconstruction Under Sample Scarcity.
GitHub repository: https://github.com/rhouhou/vae-reconstruction-scarcity
```