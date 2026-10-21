# Phase 2: PGD White-box Attack

This directory contains the Projected Gradient Descent (PGD) $L_\infty$ evasion attack on the speed-limit-sign classifier using the Adversarial Robustness Toolbox (ART).

## GPU Requirement

PGD computes gradients with backpropagation through the ResNet-50 network for 20 iterations per image.
- **GPU is strongly recommended.** The Docker container uses `tensorflow/tensorflow:2.16.2-gpu` with NVIDIA device reservation.

## Docker Usage

To generate attacked images using Docker Compose:

```bash
cd src/attack/pgd
docker compose up
```

This will run PGD against `src/modelling/runs/best_baseline.keras` on `train`, `val`, and `test` splits and output a drop-in dataset to `data/data_updated_pgd`.

### Environment Variables

In `docker-compose.yml` (or via `-e`):
- `SPLITS`: Space-separated splits to process (default: `"train val test"`)
- `SKIP_MISCLASSIFIED`: Set to `"false"` (default) to generate attacked images for all samples. Set to `"true"` to only attack clean-correct images.
- `MAX_IMAGES`: Limit images processed per split (e.g. `""` for all, or `8` for quick smoke test).
- `OVERWRITE`: Set to `"true"` to overwrite existing generated images.

## Output Structure

```text
data/data_updated_pgd/
  ├── data.yaml
  ├── train/<class>/*.png
  ├── val/<class>/*.png
  └── test/<class>/*.png

src/attack/pgd/results/generation/
  ├── generation_train_manifest.csv
  ├── generation_val_manifest.csv
  └── generation_test_manifest.csv
```

