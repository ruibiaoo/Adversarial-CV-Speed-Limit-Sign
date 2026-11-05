# Adversarial Attack Generation

This directory contains the PGD and Pixle attack scripts. Run the commands below from the **repository root** after following the installation instructions in the [root README](../../README.md).

## Prerequisites

- Clean images in `data/data_updated/{train,val,test}/<class>/` and `data/data_updated/data.yaml`.
- Baseline checkpoint at `src/modelling/runs/best_baseline.keras`. Checkpoints are not included in a fresh clone; obtain this file or train the baseline first.
- The `uv` environment installed with `~/.local/bin/uv sync --locked --python 3.12`.

The selected PGD settings in [`pgd_selected.yaml`](pgd/configs/pgd_selected.yaml) are 20 iterations, ε = 8/255, step-size ratio 0.25, and one restart. PGD is applied to RGB images in the `[0, 1]` pixel range and saves PNG files.

## Generate the attack datasets

The commands below process `train`, `val`, and `test`. Existing PGD output files may be replaced by `--overwrite`; preserve any results you need before rerunning them.

### 1. Pixle on clean images

```bash
~/.local/bin/uv run python src/attack/pixle/evaluate.py \
  --model src/modelling/runs/best_baseline.keras \
  --data data/data_updated \
  --splits train val test \
  --adversarial-dir data/data_updated_pixle \
  --results-dir src/attack/pixle/results --no-progress
```

Output: `data/data_updated_pixle/{train,val,test}/<class>/*.png` and per-split CSV/JSON reports in `src/attack/pixle/results/`.

### 2. PGD on clean images

The committed [`pgd.yaml`](pgd/configs/pgd.yaml) reads Pixle images by default. Make a temporary copy that reads clean images, leaving the committed configuration unchanged:

```bash
cp src/attack/pgd/configs/pgd.yaml /tmp/pgd_clean.yaml
sed -i 's#root: data/data_updated_pixle#root: data/data_updated#' /tmp/pgd_clean.yaml
~/.local/bin/uv run python src/attack/pgd/pgd_attacked_images_generation.py \
  --config /tmp/pgd_clean.yaml \
  --output-root data/data_updated_pgd \
  --splits train val test --overwrite
```

Output: `data/data_updated_pgd/{train,val,test}/<class>/*.png` and generation manifests in `src/attack/pgd/results/generation/`.

### 3. PGD on Pixle images (Pixle → PGD)

Run this only after Step 1. The committed `pgd.yaml` reads `data/data_updated_pixle` and writes to `data/data_pgdpixle`.

```bash
~/.local/bin/uv run python src/attack/pgd/pgd_attacked_images_generation.py \
  --splits train val test --overwrite
```

Output: `data/data_pgdpixle/{train,val,test}/<class>/*.png`. This PGD run writes to the same generation-manifest paths as Step 2, so save the earlier manifests separately if you need both sets.

## Optional: retune PGD on validation images

The selected PGD settings are already committed; this step is not required to generate images. To rerun the **36-configuration validation search** after creating the Pixle dataset:

```bash
~/.local/bin/uv run python src/attack/pgd/pgd_tuning.py
```

Results are written to `src/attack/pgd/results/pgd_tuning_results.csv`. The script does not update `pgd_selected.yaml` automatically. Do not use test images to select attack settings.

## Evaluation dataset paths

The model evaluation config, [`test_all.yaml`](../modelling/configs/test_all.yaml), currently uses the checked-in `data/data_updated/pgd_attacked` and `data/data_updated/pixle_attacked` test folders. To evaluate **newly generated** PGD and Pixle test images instead, change its `datasets.pgd_test` to `/app/data_updated_pgd/test` and `datasets.pixle_test` to `/app/data_updated_pixle/test`. Its combined-attack test path already points to `/app/data_pgdpixle/test`.

For attack-specific options and Docker alternatives, see the [PGD](pgd/README.md) and [Pixle](pixle/README.md) READMEs.
