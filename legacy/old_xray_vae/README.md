# Legacy X-ray VAE Experiment

This folder contains exploratory scripts from an earlier version of the project.

The legacy experiment trained a VAE to reconstruct biomedical X-ray images and compared downstream classification performance using:

- original images
- VAE-reconstructed images

These scripts are preserved for transparency and historical reference. They may contain hard-coded local or Colab paths and are not the recommended entry point for running the cleaned project.

The cleaned and reusable implementation should live under:

```text
src/vae_scarcity/