# Final Experiment Protocol

This document defines the final experiment plan for the cleaned repository.

The goal is to evaluate whether reconstruction-based preprocessing changes downstream X-ray classification performance under sample scarcity.

## Research Question

Does training a downstream classifier on reconstructed X-ray images improve or preserve classification performance compared with training directly on original images when labeled training data are limited?

## Compared Methods

The final experiment compares four input pipelines:

| Method | Description |
|---|---|
| Original | Classifier trained directly on original X-ray images |
| Skip VAE | Classifier trained on images reconstructed using a skip-connected VAE |
| Plain VAE | Classifier trained on images reconstructed using a plain VAE without skip connections |
| Denoising AE | Classifier trained on images reconstructed using a denoising autoencoder |

The classifier is kept fixed across methods so that the main comparison focuses on reconstruction strategy.

## Dataset

The dataset is expected locally as:

```text
data/raw/test.zip
```

Expected class folders:

```text
COVID/
NORMAL/
PNEUMONIA/
```

The dataset is not included in the repository.

## Image Preprocessing

The current preprocessing pipeline:

- loads images from a zip file
- resizes images to `64 x 64`
- converts images to RGB
- normalizes pixel values to `[0, 1]`
- creates train/validation/test splits

## Seeds

The final experiment uses five random seeds:

```text
42, 123, 777, 2024, 2025
```

Each seed should control:

- train/validation/test split
- reconstruction model initialization
- bootstrap sample selection
- classifier randomness

## Sample-Size Ratios

The downstream classifier is evaluated at these training sample-size ratios:

```text
0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00
```

## Bootstrap Repeats

Each sample-size ratio uses bootstrap resampling.

Current setting:

```text
n_bootstrap = 20
```

This can be increased later for stronger uncertainty estimates.

## Primary Metric

The primary downstream metric is:

```text
balanced accuracy
```

Balanced accuracy is used because the dataset may be class-imbalanced.

## Final Output Structure

The final multi-seed runs should be saved as:

```text
results/final_runs/
├── seed_42/
│   ├── downstream_xray_original/
│   ├── downstream_xray_skip_vae/
│   ├── downstream_xray_plain_vae/
│   └── downstream_xray_denoising_ae/
├── seed_123/
├── seed_777/
├── seed_2024/
└── seed_2025/
```

The final aggregated analysis should be saved as:

```text
results/final_analysis/
```

The final tracked figures should be copied to:

```text
results/figures/final/
```

## Planned Final Analysis

The final analysis should report:

- dataset size
- class counts
- train/validation/test counts per seed
- balanced accuracy by method and sample-size ratio
- uncertainty across seeds and bootstrap repeats
- area under the sample-size curve for each method
- method difference relative to the original-image baseline

## Interpretation Rule

No strong conclusion should be made from a single seed.

Final conclusions should only be written after all five seeds are run and aggregated.

Recommended cautious conclusion style:

> Across repeated seeds, reconstruction-based inputs did not consistently outperform the original-image baseline. Skip-connected VAE and denoising autoencoder reconstructions preserved downstream performance better than a plain VAE, suggesting that reconstruction architecture affects task-relevant information retention under sample scarcity.

The exact conclusion should be updated after the final multi-seed analysis.