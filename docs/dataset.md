# Dataset Notes

This project expects a local X-ray image dataset stored as a zip file.

The dataset is not included in this repository.

## Expected Dataset Structure

The zip file should contain one folder per class:

```text
COVID/
NORMAL/
PNEUMONIA/
```

or the class folders may be inside a root folder:

```text
dataset_root/
├── COVID/
├── NORMAL/
└── PNEUMONIA/
```

If the class folders are directly inside the zip file, use:

```yaml
root_dir: null
```

If the class folders are inside another folder, set `root_dir` to that folder name.

Example:

```yaml
root_dir: dataset_root
```

## Example Local Path

A typical local dataset path is:

```text
data/raw/test.zip
```

The `data/` folder is ignored by Git and should not be committed.

## Configuration Files

The real X-ray workflows use these config files:

```text
configs/downstream_xray_original.yaml
configs/downstream_xray_vae.yaml
```

Example configuration:

```yaml
data:
  image_size: [64, 64]
  classes:
    - COVID
    - NORMAL
    - PNEUMONIA
  root_dir: null
  test_size: 0.20
  val_size: 0.10
  normalize: true
  color_mode: rgb
```

## Preprocessing

The current pipeline:

- loads images from a zip file
- reads class labels from folder names
- resizes images to `64 x 64`
- converts images to RGB
- normalizes pixel values to `[0, 1]`
- creates reproducible train/validation/test splits

## Class Names

The class names in the config file must match the folder names inside the zip file.

For example, if the zip contains:

```text
COVID/
NORMAL/
PNEUMONIA/
```

then use:

```yaml
classes:
  - COVID
  - NORMAL
  - PNEUMONIA
```

If the zip contains:

```text
COVID19/
NORMAL/
PNEUMONIA/
```

then use:

```yaml
classes:
  - COVID19
  - NORMAL
  - PNEUMONIA
```

## How to Check the Zip Structure

To inspect the first files and folders inside the zip, run:

```bash
python - <<'PY'
import zipfile
from collections import Counter

zip_path = "data/raw/test.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    names = z.namelist()

print("First 80 entries:")
for name in names[:80]:
    print(name)

print("\nTop folder patterns:")
counter = Counter()
for name in names:
    parts = name.split("/")
    if len(parts) >= 2:
        counter["/".join(parts[:2])] += 1
    elif parts:
        counter[parts[0]] += 1

for key, count in counter.most_common(30):
    print(f"{key}: {count}")
PY
```

Use the output to set the correct `root_dir` and `classes` values in the config files.

## Running the Original X-ray Baseline

```bash
python scripts/run_downstream_xray_original.py \
  --config configs/downstream_xray_original.yaml \
  --data-zip data/raw/test.zip \
  --output-dir results/downstream_xray_original
```

## Running the VAE-Reconstructed X-ray Pipeline

This workflow requires TensorFlow:

```bash
python -m pip install -e ".[deep-learning]"
```

Then run:

```bash
python scripts/run_downstream_xray_reconstruction.py \
  --config configs/downstream_xray_vae.yaml \
  --data-zip data/raw/test.zip \
  --output-dir results/downstream_xray_vae
```

## Important Notes

Datasets are not included in this repository because they may be large and may have their own licenses or terms of use.

Users should download the dataset separately and place it locally under:

```text
data/raw/
```

Do not commit datasets, zip files, model checkpoints, or generated experiment outputs to GitHub.