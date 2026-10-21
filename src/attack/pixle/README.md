# Phase 2: Pixle Black-box Attack

This directory attacks the speed-limit-sign classifier using Pixle, an $L_0$ black-box attack based on rearranging pixels. The attack only requires a probability oracle and does not require gradients or model weights.

## GPU Execution & Multi-Threading

- **GPU-Accelerated**: The Docker container uses `tensorflow/tensorflow:2.16.2-gpu` with NVIDIA GPU device reservation.
- **8 Workers Multi-Threading**: Pixle runs 8 parallel worker threads (`--workers 8`, configurable via `WORKERS` env var) that attack multiple images concurrently.

## Docker Usage

To generate attacked images using Docker Compose:

```bash
cd src/attack/pixle
docker compose up
```

This runs unit tests first, then generates attacked images for `train`, `val`, and `test` splits against `src/modelling/runs/best_baseline.keras`.

### Environment Variables

In `docker-compose.yml` (or via `-e`):
- `SPLITS`: Space-separated splits to process (default: `"train val test"`)
- `SKIP_MISCLASSIFIED`: Set to `"false"` (default) to generate images for all samples. Set to `"true"` to skip generating images that the model already misclassifies.
- `WORKERS`: Number of parallel worker threads (default: `"8"`).
- `MAX_IMAGES`: Limit images processed per split (e.g. `""` for all, or `8` for quick smoke test).

## Output Structure

```text
data/data_updated_pixle/
  ├── data.yaml
  ├── train/<class>/*.png
  ├── val/<class>/*.png
  └── test/<class>/*.png

src/attack/pixle/results/
  ├── train/
  │   ├── pixle_results.csv
  │   └── pixle_metrics.json
  ├── val/
  │   ├── pixle_results.csv
  │   └── pixle_metrics.json
  └── test/
      ├── pixle_results.csv
      └── pixle_metrics.json
```
