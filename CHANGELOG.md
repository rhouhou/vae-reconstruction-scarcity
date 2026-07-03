# Changelog

## v0.1.0

Initial stable release of the cleaned VAE Reconstruction Under Sample Scarcity project.

### Added

- Original-image downstream baseline
- Skip-connected VAE reconstruction pipeline
- Plain VAE reconstruction pipeline
- Denoising autoencoder reconstruction pipeline
- Final five-seed experiment protocol
- Final experiment runner
- Multi-seed aggregation script
- Sample-size efficiency analysis
- Final result figures
- Dataset documentation
- Experiment workflow documentation
- Final protocol documentation
- Results interpretation documentation
- GitHub Actions CI
- MIT License
- Citation metadata

### Results

Across five random seeds, reconstruction-based inputs did not outperform the original-image baseline.

Skip VAE and denoising autoencoder reconstructions preserved downstream performance relatively well, while plain VAE reconstruction substantially degraded downstream performance.

The sample-size analysis showed that Original, Skip VAE, and Denoising AE reached more than 99% of their full-data performance at 60% of the training split.

### Notes

The dataset, trained model checkpoints, and raw generated result CSV files are not included in the repository.

This project is for research, education, and portfolio demonstration only. It is not intended for clinical diagnosis or deployment in healthcare settings.