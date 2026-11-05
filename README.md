# Defending Speed-Limit Sign Classification Against Adversarial Attacks

## Project Overview

This DSA4266 project uses a ResNet50 speed-limit sign classifier to evaluate PGD and Pixle attacks. It trains and compares input-level, training-stage and output-level defences, then generates Grad-CAM and Ablation-CAM visualisations.

## Installation

Use Ubuntu/Linux and Python 3.12. `pyproject.toml` and `uv.lock` pin the local environment (TensorFlow 2.16.2 and `tf-keras` 2.16.0). The Docker training/evaluation service requires Docker Compose, NVIDIA drivers, a GPU and the NVIDIA Container Toolkit. Local attack scripts can run on CPU, slowly.

```bash
git clone https://github.com/ruibiaoo/Adversarial-CV-Speed-Limit-Sign.git
cd Adversarial-CV-Speed-Limit-Sign
git fetch origin clean-up
git switch --track origin/clean-up
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv sync --locked --python 3.12
~/.local/bin/uv run --locked python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
```

Run commands from the repository root unless stated otherwise. `[]` means local TensorFlow does not detect a GPU; Docker GPU access must be checked separately.

## Dataset Sources and Preparation

Original sources: [Kaggle — Speed Limit Dataset USA](https://www.kaggle.com/datasets/basiradaqiq/speed-limit-dataset-usa) and [Hugging Face — Speed Limit Signs](https://huggingface.co/datasets/ctacke/speed-limit-signs). Download them manually **only if rebuilding from raw data**.

For the runnable pipeline, use the checked-in `data/data_updated/`: seven classes (`10_MPH`–`70_MPH`), 1,080 training, 220 validation and 115 test images. The layout is `data/data_updated/{train,val,test}/<class>/` plus `data.yaml`. Model loaders use RGB and resize images to 256 × 256.

The optional cropper requires a manually assembled, flat YOLO dataset at `data/data_raw/` with `images/`, matching `labels/`, and `data.yaml`:

```bash
~/.local/bin/uv run --locked python src/data/prepare_classification_data.py \
  --raw-dir data/data_raw --output-dir data/rebuilt_from_raw --seed 4266
```

Output: `data/rebuilt_from_raw/{train,val,test}/` and `data.yaml`. **This does not recreate the curated split.** The repository lacks the raw-source conversion, merge, filtering and deduplication steps. Use `data/data_updated/` for the workflow below.

## Quick Start — Evaluate Existing Models

The checked-in test sets are in `data/data_updated/` and `data/data_pgdpixle/`. Evaluation requires these six checkpoints in `src/modelling/runs/`:

| File | Model/use |
| --- | --- |
| `best_baseline.keras` | M0 baseline |
| `grl.keras` | M1 Gradient Reversal Layer |
| `best_adversarial_retrained.keras` | M2 Adversarial Retraining |
| `best_robust.keras` | M3 Adversarial Objective Training |
| `best_convae_rn50.keras` | M4 Fine-tuning with ConvAE |
| `convae_adv2clean_aug_v6.keras` | ConvAE input preprocessing |

**Most `.keras` files are not included in Git.** Only `best_baseline.keras` exists in the checkout inspected here; a fresh clone may have none. Obtain missing checkpoints from the authors or train them below. `test_all.py` checks all configured model paths before running.

**Run:**

```bash
docker compose -f src/modelling/docker-compose.yml build
docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier python -u test_all.py
```

**Output:** `src/modelling/final_final_results_i_promise/tests/` and summary files. To evaluate fewer models, remove unavailable entries from `models` in `src/modelling/configs/test_all.yaml`; if ConvAE is missing, also remove its preprocessing entry/chains or disable preprocessing.

## Full Reproduction — Train and Evaluate Everything

Start from the checked-in curated dataset. The training YAML currently describes M1, so edit it before each model run. Back up any existing attacked data or weights: regeneration and the checkpoint-copy commands can overwrite files.

### Step 1 — Prepare the clean dataset

Use the checked-in clean train/validation/test split. Raw-source rebuilding is described above but cannot reproduce this exact split.

**Configuration:** `data/data_updated/data.yaml` and its seven class folders per split.

**Run:**

```bash
test -f data/data_updated/data.yaml
test -d data/data_updated/train && test -d data/data_updated/val && test -d data/data_updated/test
```

**Output:** `data/data_updated/{train,val,test}/` is ready.

### Step 2 — Train baseline ResNet50 (M0)

Edit `src/modelling/configs/train.yaml` before training:

| Setting | Value |
| --- | --- |
| `data` / `project` | `"/app/eval_data"` / `"/app/runs"` |
| `pretrained` | `true` (may download ImageNet weights) |
| `grl.enabled` / `convae.enabled` / `robust` | `false` / `false` / `false` |

Keep `model_name: resnet50` and `image_size: 256`. `src/modelling/configs/tuner.yaml` already has `tuner: false`. `train.py` reads `train.yaml` directly.

**Run:**

```bash
docker compose -f src/modelling/docker-compose.yml run --build --rm phase1-classifier python -u train.py
cp src/modelling/runs/best.keras src/modelling/runs/best_baseline.keras
```

**Output:** `src/modelling/runs/best_baseline.keras`.

### Step 3 — Generate PGD, Pixle and Pixle → PGD datasets

Generate clean-source PGD and Pixle first, then apply PGD to Pixle. `src/attack/pgd/configs/pgd_selected.yaml` fixes 20 iterations, ε=8/255, step ratio 0.25 and one restart. The committed `pgd.yaml` points to Pixle input, so the first PGD command uses a temporary clean-source copy.

**Configuration:**

| Attack | Input | Output |
| --- | --- | --- |
| PGD | `data/data_updated/` | `data/data_updated_pgd/` |
| Pixle | `data/data_updated/` | `data/data_updated_pixle/` |
| Pixle → PGD | `data/data_updated_pixle/` | `data/data_pgdpixle/` |

**Run in this order:**

```bash
cp src/attack/pgd/configs/pgd.yaml /tmp/pgd_clean.yaml
sed -i 's#root: data/data_updated_pixle#root: data/data_updated#' /tmp/pgd_clean.yaml
~/.local/bin/uv run --locked python src/attack/pgd/pgd_attacked_images_generation.py \
  --config /tmp/pgd_clean.yaml --output-root data/data_updated_pgd \
  --splits train val test --overwrite

~/.local/bin/uv run --locked python src/attack/pixle/evaluate.py \
  --model src/modelling/runs/best_baseline.keras --data data/data_updated \
  --splits train val test --adversarial-dir data/data_updated_pixle \
  --results-dir src/attack/pixle/results --no-progress

~/.local/bin/uv run --locked python src/attack/pgd/pgd_attacked_images_generation.py \
  --splits train val test --overwrite
```

**Output:** Class-organised images in the three output roots above; generation reports under `src/attack/pgd/results/` and `src/attack/pixle/results/`.

### Step 4 — Train ConvAE and training-stage models

Train ConvAE, build the combined dataset, then train M1–M4. In `src/modelling/configs/train_convae.yaml`, set `data: "/app/eval_data"` for clean targets; keep `adv_data: ["/app/data_pgdpixle"]` and the existing `project`/`name`.

**Run ConvAE and combine data:**

```bash
docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier python -u train_convae.py
cp src/modelling/runs/convae/convae_adv2clean_aug_v6/best.keras \
  src/modelling/runs/convae_adv2clean_aug_v6.keras
~/.local/bin/uv run --locked python src/data/combine_datasets.py
```

**Output:** ConvAE files in `src/modelling/runs/` and `data/data_combined/{train,val,test}/`. The combine script copies into that directory without removing stale files; check its contents before rebuilding.

For each model, edit `src/modelling/configs/train.yaml` to `data: "/app/data"`, `project: "/app/runs"`, and `pretrained: "/app/runs/best_baseline.keras"`. Set these switches; keep other training settings unchanged:

| Model | `grl.enabled` | `robust` | `convae.enabled` | Final checkpoint |
| --- | --- | --- | --- | --- |
| M1: GRL | `true` | `false` | `false` | `grl.keras` |
| M2: Adversarial Retraining | `false` | `false` | `false` | `best_adversarial_retrained.keras` |
| M3: Adversarial Objective Training | `false` | `true` | `false` | `best_robust.keras` |
| M4: Fine-tuning with ConvAE | `false` | `false` | `true` | `best_convae_rn50.keras` |

For M3 keep `robust_norm: l2`, `robust_eps: 3.0`, `robust_step_size: 0.5`, `robust_steps: 7`. For M4 keep `convae.model_path: "/app/runs/convae/convae_adv2clean_aug_v6/best.keras"` and `convae.trainable: false`.

**Run once per row, after updating the YAML:**

```bash
docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier python -u train.py
```

**Output:** M1 saves `src/modelling/runs/grl.keras`. Copy each later model's `best.keras` immediately after its run, before the next model overwrites it.

After M2:

```bash
cp src/modelling/runs/best.keras src/modelling/runs/best_adversarial_retrained.keras
```

After M3:

```bash
cp src/modelling/runs/best.keras src/modelling/runs/best_robust.keras
```

After M4:

```bash
cp src/modelling/runs/best.keras src/modelling/runs/best_convae_rn50.keras
```

### Step 5 — Evaluate input-level, output-level and combined defences

Tune standalone input/output defences on validation data and evaluate their chains. `test_all.py` also evaluates all five models, the ConvAE input chain, output chains and no-defence baselines on four test sets.

**Configuration:** The standalone scripts need `data/data_clean_pgd_pixle/`. Its creation command refuses an existing target; back it up before using `--overwrite`.

**Run standalone defences:**

```bash
~/.local/bin/uv run --locked python src/data/combine_clean_pgd_pixle.py
cd src/defenses
MODEL=../../src/modelling/runs/best_baseline.keras
DATA=../../data/data_clean_pgd_pixle
~/.local/bin/uv run --locked python modules/preprocessing/tune_preprocessing.py --model "$MODEL" --data "$DATA"
~/.local/bin/uv run --locked python modules/preprocessing/test_defences.py --model "$MODEL" --data "$DATA"
~/.local/bin/uv run --locked python modules/post-processing/tune_postprocessing.py --model "$MODEL" --data "$DATA"
~/.local/bin/uv run --locked python modules/post-processing/test_postprocessing.py --model "$MODEL" --data "$DATA"
cd ../..
```

**Output:** `src/defenses/tuning_logs/`, tuned YAMLs under `src/defenses/modules/`, and `src/defenses/testing_results/`.

For *newly generated* PGD/Pixle attacks, change these values in `src/modelling/configs/test_all.yaml`; its committed values use older checked-in attack test sets:

| `datasets` key | New value |
| --- | --- |
| `pgd_test` | `"/app/data_updated_pgd/test"` |
| `pixle_test` | `"/app/data_updated_pixle/test"` |

Keep `clean_test` and `pgdpixle_test` as configured. Check that all six required checkpoints exist. The configured chains include ConvAE → JPEG, and `include_none: true` provides model-only results.

**Run combined evaluation:**

```bash
docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier python -u test_all.py
```

**Output:** `src/modelling/final_final_results_i_promise/tests/<combination>/<test-set>/` and summary files. Set a different `output_dir` in `test_all.yaml` first if preserving existing results.

### Step 6 — Generate Grad-CAM and Ablation-CAM

Generate matched visualisations for M0, M0 + ConvAE → JPEG + High-Confidence Filtering, and M4.

**Configuration:** Use a PGD filename present in the matching `predictions.csv`; model, data and CSV must come from the same evaluation run.

**Run:** From the repository root:

```bash
IMAGE='20s1_png.rf.21e5ce2e19a562face23463639c38966.png'
RESULTS=/app/final_final_results_i_promise/tests
cam_pair() {
  for CAM in gradcam ablationcam; do
    CAM_DIR="$CAM""_20mph"
    docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier \
      python -u "$CAM.py" --model "$1" --predictions "$2" \
      --image "$IMAGE" --image-size 256 --target true \
      --preprocessing "$3" --postprocessing "$4" \
      --output-dir "/app/results/$CAM_DIR/$5"
  done
}
cam_pair /app/runs/best_baseline.keras \
  "$RESULTS/pre_none__model_baseline__post_none/pgd_test/predictions.csv" \
  none none baseline
cam_pair /app/runs/best_baseline.keras \
  "$RESULTS/pre_convae_then_jpeg__model_baseline__post_highconf/pgd_test/predictions.csv" \
  convae_jpeg highconf convae_jpeg_highconf
cam_pair /app/runs/best_convae_rn50.keras \
  "$RESULTS/pre_none__model_convae_rn50__post_none/pgd_test/predictions.csv" \
  none none fine_tuning_convae
```

**Output:** `src/modelling/results/gradcam_20mph/` and `src/modelling/results/ablationcam_20mph/`.

Optional, after Pixle generation: run the validation-only PGD search with `~/.local/bin/uv run --locked python src/attack/pgd/pgd_tuning.py`. It writes `src/attack/pgd/results/pgd_tuning_results.csv`; update `pgd_selected.yaml` manually if selecting different parameters. Never tune on test data.

## Model Interpretability

Step 6 uses a 20 MPH PGD image present in the checked-in and regenerated test folders. Each CAM script reads the image path from its matching `predictions.csv`; `--target true` explains the actual 20 MPH class. Use another exact CSV filename for `IMAGE`, and update `RESULTS` if you changed `test_all.yaml`'s `output_dir`. Never mix checkpoints, images and CSVs from different evaluation runs.

## Repository Structure

```text
.
├── data/
│   ├── data_updated/              Clean data and checked-in attacked test sets
│   ├── data_updated_pgd/          PGD train/val/test
│   ├── data_updated_pixle/        Pixle train/val/test
│   ├── data_pgdpixle/             Pixle → PGD train/val/test
│   └── data_combined/             Combined training data
├── src/
│   ├── data/                      Dataset preparation/combination
│   ├── attack/
│   │   ├── pgd/                   PGD scripts and configs
│   │   └── pixle/                 Pixle scripts
│   ├── modelling/                 Training, evaluation, CAMs, configs, runs
│   └── defenses/modules/          Input/output defences
├── pyproject.toml
└── uv.lock
```

## Troubleshooting

- **Missing checkpoints:** `.keras` files are ignored by Git. Obtain them or finish Steps 2 and 4; `test_all.py` checks every configured path.
- **Wrong paths:** Run root commands from the repository root; only the standalone defence block uses `cd src/defenses`. Docker mounts clean data at `/app/eval_data` and combined data at `/app/data`.
- **Missing raw data:** The optional cropper needs YOLO `images/`, `labels/` and `data.yaml`. The checked-in curated data is sufficient for the main workflow.
- **GPU/memory:** Check NVIDIA driver/container setup. Reduce `batch` in the relevant YAML or Pixle `--workers` if necessary and record the change.
