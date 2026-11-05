# Defending Against Adversarial Attacks on Speed-Limit Sign Classification

This DSA4266 project evaluates PGD and Pixle attacks against a ResNet50 speed-limit sign classifier. It trains and compares input-level, training-stage, and output-level defences and generates Grad-CAM and Ablation-CAM visualisations.

Run commands from the **repository root** unless stated otherwise. This guide targets `clean-up`: the default branch contains the principal scripts, but its configurations may differ.

## Repository structure

```text
data/
  data_updated/                 Clean train/val/test; checked-in attacked test sets
  data_updated_pgd/             PGD train/val/test
  data_updated_pixle/           Pixle train/val/test
  data_pgdpixle/                Pixle -> PGD train/val/test
  data_combined/                Four-source training dataset
src/
  data/
    prepare_classification_data.py   Crop/split YOLO-labelled images
    combine_datasets.py              Build data_combined/
    combine_clean_pgd_pixle.py       Build standalone-defence test layout
  attack/
    pgd/{pgd_tuning.py,pgd_attacked_images_generation.py,configs/}
    pixle/evaluate.py
  modelling/
    {train.py,train_convae.py,test_all.py,gradcam.py,ablationcam.py}
    configs/{train.yaml,train_convae.yaml,test_all.yaml}
    runs/                            Local checkpoints (not committed)
  defenses/modules/
    preprocessing/                 JPEG, median, randomised smoothing
    post-processing/               Reverse sigmoid, high-confidence, noise
pyproject.toml, uv.lock             Local Python dependencies
```

## Requirements and installation

Use Ubuntu/Linux and Python **3.12**. The root lockfile includes TensorFlow **2.16.2** and `tf-keras` **2.16.0**. Docker images install separate `requirements.txt` files. The Compose services below request an NVIDIA GPU; install Docker Engine, Compose, NVIDIA drivers, and the NVIDIA Container Toolkit to use them. Local Python commands can run on CPU but may be slow. W&B is optional; no API key or `.env` is needed for local runs.

```bash
git clone https://github.com/ruibiaoo/Adversarial-CV-Speed-Limit-Sign.git
cd Adversarial-CV-Speed-Limit-Sign
git fetch origin clean-up
git switch --track origin/clean-up
curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv sync --locked --python 3.12
~/.local/bin/uv run --locked python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
```

An empty `[]` means local TensorFlow did not detect a GPU; see [TensorFlow's GPU guide](https://www.tensorflow.org/install/pip). The installer command follows the [official uv guide](https://docs.astral.sh/uv/getting-started/installation/).

## Dataset sources and preparation

Original sources: [Kaggle — Speed Limit Dataset USA](https://www.kaggle.com/datasets/basiradaqiq/speed-limit-dataset-usa) and [Hugging Face — Speed Limit Signs](https://huggingface.co/datasets/ctacke/speed-limit-signs). Download from those pages manually if rebuilding the raw data; this repository has no downloader or converter for both source formats. The checked-in curated `data/data_updated/` has seven classes (`10_MPH` through `70_MPH`) and **1,415 images**: 1,080 train, 220 validation, and 115 test.

For **running the project**, use the checked-in `data/data_updated/{train,val,test}/<class>/` and `data/data_updated/data.yaml`. No raw download is required. Model loaders convert images to RGB and resize to 256 × 256. Do not overwrite this curated split with the generic preparer.

For **raw preparation**, keep manually downloaded archives in a staging directory such as `data/raw_sources/`. `src/data/prepare_classification_data.py` requires a *flat YOLO detection* input, assembled manually:

```text
data/data_raw/
  images/<stem>.jpg
  labels/<stem>.txt       # YOLO bounding boxes/polygons
  data.yaml               # class ID -> class name
```

It crops each labelled sign, converts to RGB, optionally resizes with `--target-size WIDTH HEIGHT`, groups related augmented variants, and writes class-organised splits plus `data.yaml`. Given a separately assembled seven-class YOLO input:

```bash
~/.local/bin/uv run --locked python src/data/prepare_classification_data.py \
  --raw-dir data/data_raw --output-dir data/rebuilt_from_raw --seed 4266
```

Output: `data/rebuilt_from_raw/{train,val,test}/<class>/` and `data/rebuilt_from_raw/data.yaml`. **This does not reconstruct the curated split:** this branch has no script for merging the two downloads, converting the Hugging Face image-folder layout to YOLO, filtering to seven classes, or deduplicating. The generic preparer's default split rules also differ from the checked-in 1,080/220/115 allocation. Use `data/data_updated/` for the pipeline below.

## Quick start — existing checkpoints

**Purpose:** Evaluate the checked-in clean/adversarial test sets without retraining. **Prerequisites:** Complete installation above, retain `data/data_updated/` and `data/data_pgdpixle/`, and place these checkpoints in `src/modelling/runs/`:

```text
best_baseline.keras
best_adversarial_retrained.keras
best_robust.keras
grl.keras
best_convae_rn50.keras
convae_adv2clean_aug_v6.keras
```

`.keras` files are Git-ignored; a fresh clone does **not** include the weights. The checkout inspected for this README has `best_baseline.keras` only. Obtain the other checkpoints from the authors or train them below. The default evaluation config uses the checked-in `data/data_updated/{test,pgd_attacked,pixle_attacked}` and `data/data_pgdpixle/test`; similarly named regenerated attack sets are not necessarily identical.

**Command:**

```bash
docker compose -f src/modelling/docker-compose.yml build
docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier python -u test_all.py
```

**Output:** `src/modelling/final_final_results_i_promise/tests/<combination>/<test-set>/` (predictions and reports) plus top-level summary files. With only M0, edit `src/modelling/configs/test_all.yaml` to keep only `models.baseline` and disable/remove ConvAE preprocessing; otherwise `test_all.py` checks for every configured checkpoint. CAM commands are in Step 11.

## Full reproduction — from the curated split

These are the executable stages **after** raw-data curation. The committed YAMLs describe one run, not five immutable model recipes. Make the specified YAML edits before each command and preserve a copy of each version if exact run provenance matters. Never choose attack or defence settings using test results.

### 1. Prepare the clean dataset

**Purpose:** Supply seven-class clean train/val/test folders. **Prerequisites:** Clone the checked-in dataset; optional raw rebuilding first requires the manually assembled YOLO layout above.

**Command:**

```bash
test -f data/data_updated/data.yaml
test -d data/data_updated/train && test -d data/data_updated/val && test -d data/data_updated/test
```

**Output:** Use checked-in `data/data_updated/{train,val,test}` downstream. The optional raw-source command in the dataset section writes only `data/rebuilt_from_raw/`.

### 2. Train baseline ResNet50 (M0)

**Purpose:** Train the seven-class baseline. **Prerequisites:** Clean train/val set and Docker GPU. Edit `src/modelling/configs/train.yaml`: `data: "/app/eval_data"`, `project: "/app/runs"`, `pretrained: true`, `grl.enabled: false`, `convae.enabled: false`, `robust: false`. Keep `model_name: resnet50`, `image_size: 256`, `batch: 16`; `configs/tuner.yaml` already has `tuner: false`. `train.py` reads `train.yaml` directly and has no config flag. ImageNet weights may download on first use.

**Command:**

```bash
docker compose -f src/modelling/docker-compose.yml run --build --rm phase1-classifier python -u train.py
cp src/modelling/runs/best.keras src/modelling/runs/best_baseline.keras
```

**Output:** `src/modelling/runs/best_baseline.keras`, copied from the best-validation checkpoint. `cp` overwrites a same-named file; back it up first if needed.

### 3. Generate clean-source PGD images

**Purpose:** Create PGD train/val/test data. **Prerequisites:** M0 and the clean split. `src/attack/pgd/configs/pgd_selected.yaml` specifies 20 iterations, ε=8/255, step ratio 0.25, and one restart. The committed `pgd.yaml` targets **Pixle** data for the combined attack, so make a temporary clean-source config. The generator always reads the selected parameters from `pgd_selected.yaml`.

**Command:**

```bash
cp src/attack/pgd/configs/pgd.yaml /tmp/pgd_clean.yaml
sed -i 's#root: data/data_updated_pixle#root: data/data_updated#' /tmp/pgd_clean.yaml
~/.local/bin/uv run --locked python src/attack/pgd/pgd_attacked_images_generation.py \
  --config /tmp/pgd_clean.yaml --output-root data/data_updated_pgd \
  --splits train val test --overwrite
```

**Output:** `data/data_updated_pgd/{train,val,test}/<class>/*.png`, `data.yaml`, and `src/attack/pgd/results/generation/` manifests. `--overwrite` replaces matching outputs; back up existing attacked folders first.

### 4. Generate Pixle images

**Purpose:** Create Pixle train/val/test data. **Prerequisites:** M0 and clean split. `evaluate.py` defaults to 100 restarts, 50 iterations, patch size 3, random mapping, restart-iterative search, overwrite operation, and seed 4266.

**Command:**

```bash
~/.local/bin/uv run --locked python src/attack/pixle/evaluate.py \
  --model src/modelling/runs/best_baseline.keras --data data/data_updated \
  --splits train val test --adversarial-dir data/data_updated_pixle \
  --results-dir src/attack/pixle/results --no-progress
```

**Output:** `data/data_updated_pixle/{train,val,test}/<class>/` and `src/attack/pixle/results/`. Same-named output files are replaced; back up existing outputs first.

### 5. Generate combined images: Pixle → PGD

**Purpose:** Apply PGD to Step 4's Pixle images. **Prerequisites:** M0, Pixle train/val/test, and selected PGD YAML. Committed `pgd.yaml` points to `data/data_updated_pixle` and outputs to `data/data_pgdpixle`. The optional sweep uses **Pixle validation**, not test, and writes 36 evaluations; inspect them and manually edit `pgd_selected.yaml` only if repeating parameter selection.

**Command:**

```bash
~/.local/bin/uv run --locked python src/attack/pgd/pgd_attacked_images_generation.py \
  --splits train val test --overwrite
```

**Output:** PNGs under `data/data_pgdpixle/{train,val,test}/<class>/` and manifests under `src/attack/pgd/results/generation/`. To repeat the 36-configuration validation-only search *before generation*, run `~/.local/bin/uv run --locked python src/attack/pgd/pgd_tuning.py` after Step 4. It writes CSV/plots under `src/attack/pgd/results/` and does **not** update `pgd_selected.yaml` automatically.

### 6. Train denoising ConvAE

**Purpose:** Train Pixle→PGD-to-clean reconstruction. **Prerequisites:** Clean and combined-attack train/val sets. In `src/modelling/configs/train_convae.yaml`, change `data: "/app/data"` to `data: "/app/eval_data"`: `/app/data` is the filename-prefixed combined set, not the clean targets expected by the pairing code. Keep `adv_data: ["/app/data_pgdpixle"]`, `project: "/app/runs/convae"`, and `name: "convae_adv2clean_aug_v6"`.

**Command:**

```bash
docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier python -u train_convae.py
cp src/modelling/runs/convae/convae_adv2clean_aug_v6/best.keras \
  src/modelling/runs/convae_adv2clean_aug_v6.keras
```

**Output:** Nested `src/modelling/runs/convae/convae_adv2clean_aug_v6/best.keras`; the root-level copy is required by `test_all.yaml` for ConvAE input preprocessing.

### 7. Run input-level and output-level defences

**Purpose:** Tune on validation, then test JPEG, median, randomised smoothing, reverse sigmoid, high-confidence filtering and Gaussian score noise (including ordered chains). **Prerequisites:** M0 and clean/PGD/Pixle train/val/test data. ConvAE preprocessing is run in Step 10's `test_all.py`, not the standalone preprocessing test script.

**Command:**

```bash
~/.local/bin/uv run --locked python src/data/combine_clean_pgd_pixle.py
cd src/defenses
~/.local/bin/uv run --locked python modules/preprocessing/tune_preprocessing.py \
  --model ../../src/modelling/runs/best_baseline.keras --data ../../data/data_clean_pgd_pixle
~/.local/bin/uv run --locked python modules/preprocessing/test_defences.py \
  --model ../../src/modelling/runs/best_baseline.keras --data ../../data/data_clean_pgd_pixle
~/.local/bin/uv run --locked python modules/post-processing/tune_postprocessing.py \
  --model ../../src/modelling/runs/best_baseline.keras --data ../../data/data_clean_pgd_pixle
~/.local/bin/uv run --locked python modules/post-processing/test_postprocessing.py \
  --model ../../src/modelling/runs/best_baseline.keras --data ../../data/data_clean_pgd_pixle
cd ../..
```

**Output:** `data/data_clean_pgd_pixle/`; tuning grids/settings in `src/defenses/tuning_logs/` and `src/defenses/modules/*/tuned_defences.yaml`; test reports in `src/defenses/testing_results/`. The combine script refuses an existing target; run it once or back up the target before using its destructive `--overwrite` flag.

### 8. Train training-stage models (M1–M4)

**Purpose:** Train GRL, adversarial-retrained, adversarial-objective and ConvAE-fine-tuned classifiers. **Prerequisites:** M0, all four train/val datasets, and the ConvAE checkpoint for M4. First build the dataset mounted by modelling Docker as `/app/data`.

**Command:**

```bash
~/.local/bin/uv run --locked python src/data/combine_datasets.py
```

**Output:** `data/data_combined/{train,val,test}/<class>/`. It copies into the target without removing stale files; use a fresh target for a clean rebuild.

For **each row**, edit `src/modelling/configs/train.yaml` to `data: "/app/data"`, `project: "/app/runs"`, `pretrained: "/app/runs/best_baseline.keras"` and the listed switches. Retain the file's optimizer, augmentation, epoch, image-size and inner-training settings unless shown. Run `train.py` after each edit; copy its output **before** moving to the next row. `name` is a tracking label, not the checkpoint filename.

| Model | `grl.enabled` | `robust` | `convae.enabled` | Additional requirement / output |
| --- | --- | --- | --- | --- |
| M1 — Gradient Reversal Layer | `true` | `false` | `false` | Four domains in combined data; `runs/grl.keras`, plus `best_classifier.keras` export |
| M2 — Adversarial Retraining | `false` | `false` | `false` | Copy `runs/best.keras` to `best_adversarial_retrained.keras` |
| M3 — Adversarial Objective Training | `false` | `true` | `false` | Current YAML uses L2 PGD (`robust_norm: l2`, ε=3.0, step=0.5, 7 steps); copy to `best_robust.keras` |
| M4 — Fine-tuning with ConvAE | `false` | `false` | `true` | Keep `convae.model_path: "/app/runs/convae/convae_adv2clean_aug_v6/best.keras"`, `convae.trainable: false`; copy to `best_convae_rn50.keras` |

**Command:** Run the training command once for **each** row after making that row's YAML edits:

```bash
docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier python -u train.py
```

After **M2** finishes, run:

```bash
cp src/modelling/runs/best.keras src/modelling/runs/best_adversarial_retrained.keras
```

After **M3** finishes, run:

```bash
cp src/modelling/runs/best.keras src/modelling/runs/best_robust.keras
```

After **M4** finishes, run:

```bash
cp src/modelling/runs/best.keras src/modelling/runs/best_convae_rn50.keras
```

**Output:** Named M0–M4 checkpoints in `src/modelling/runs/`. M1 writes `grl.keras` directly. Copy each `best.keras` immediately after its own run, because the next run overwrites it.

### 9. Evaluate M0–M4 without test-time defences

**Purpose:** Score each model on clean, PGD, Pixle and Pixle→PGD test sets. **Prerequisites:** All five model checkpoints and four test sets. For *freshly regenerated* attacks, edit `src/modelling/configs/test_all.yaml`: `datasets.pgd_test: "/app/data_updated_pgd/test"`, `datasets.pixle_test: "/app/data_updated_pixle/test"`. The committed paths instead select the older `data_updated/pgd_attacked` and `pixle_attacked` folders. Temporarily set `preprocessing.enabled: false`, `postprocessing.enabled: false`, and `output_dir: "/app/results/models_only"` to keep this evaluation separate from the combined-defence run.

**Command:**

```bash
docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier python -u test_all.py
```

**Output:** `src/modelling/results/models_only/tests/pre_none__model_<model>__post_none/<test-set>/` plus top-level summaries.

### 10. Evaluate combined defences

**Purpose:** Test configured orders of input preprocessing, M0–M4, and output postprocessing. **Prerequisites:** Step 9 inputs and `src/modelling/runs/convae_adv2clean_aug_v6.keras`. Restore `preprocessing.enabled: true`, `postprocessing.enabled: true`, and `output_dir: "/app/final_final_results_i_promise"` in `test_all.yaml`. Its committed input mode is `explicit` (including ConvAE→JPEG and JPEG→median→randomised smoothing); output mode is `permutations` of reverse sigmoid, Gaussian score noise and high-confidence filtering.

**Command:**

```bash
docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier python -u test_all.py
```

**Output:** `src/modelling/final_final_results_i_promise/tests/<combination>/<test-set>/` and aggregate files. This path is also used by the Step 11 CAM commands.

### 11. Generate Grad-CAM and Ablation-CAM

**Purpose:** Explain the same true-class 20 MPH PGD image across M0, M0 + ConvAE→JPEG + high-confidence filtering, and M4. **Prerequisites:** Matching checkpoints and `test_all.py` `predictions.csv` files. The example filename `20s1_png.rf.21e5ce2e19a562face23463639c38966.png` exists in the checked-in `data/data_updated/pgd_attacked/20_MPH/` and `data/data_updated_pgd/test/20_MPH/`. Use the PGD dataset selected in `test_all.yaml` when you generated the CSVs; do not combine a CSV from one attack set with an image from another. `--target true` fixes the CAM target to 20 MPH. Postprocessing verifies the CSV prediction; the CAM explains the classifier score before postprocessing.

**Command:**

```bash
IMAGE='20s1_png.rf.21e5ce2e19a562face23463639c38966.png'
for CAM in gradcam ablationcam; do
  docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier \
    python -u "${CAM}.py" --model /app/runs/best_baseline.keras \
    --predictions /app/final_final_results_i_promise/tests/pre_none__model_baseline__post_none/pgd_test/predictions.csv \
    --image "$IMAGE" --image-size 256 --target true \
    --output-dir "/app/results/${CAM}_20mph/baseline"
  docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier \
    python -u "${CAM}.py" --model /app/runs/best_baseline.keras \
    --predictions /app/final_final_results_i_promise/tests/pre_convae_then_jpeg__model_baseline__post_highconf/pgd_test/predictions.csv \
    --preprocessing convae_jpeg --postprocessing highconf \
    --image "$IMAGE" --image-size 256 --target true \
    --output-dir "/app/results/${CAM}_20mph/convae_jpeg_highconf"
  docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier \
    python -u "${CAM}.py" --model /app/runs/best_convae_rn50.keras \
    --predictions /app/final_final_results_i_promise/tests/pre_none__model_convae_rn50__post_none/pgd_test/predictions.csv \
    --image "$IMAGE" --image-size 256 --target true \
    --output-dir "/app/results/${CAM}_20mph/fine_tuning_convae"
done
```

**Output:** Figures under `src/modelling/results/gradcam_20mph/` and `src/modelling/results/ablationcam_20mph/`. Both scripts check the checkpoint against saved CSV probabilities; model, preprocessing, image and CSV must come from the **same** evaluation run.

## Troubleshooting and known gaps

- **Missing datasets:** Check `pwd` and `ls data/data_updated/{train,val,test}`. Do not pass a Hugging Face image-folder download directly to the YOLO cropper.
- **Missing checkpoints:** Git ignores `.keras`. Obtain weights from the authors or train them; `test_all.py` validates every configured model path and the ConvAE used by an enabled input chain.
- **Wrong paths:** Docker mounts `/app/eval_data` as clean data, `/app/data` as `data_combined`, and `/app/data_pgdpixle` as Pixle→PGD. ConvAE training needs `/app/eval_data` as its clean target. Run standalone defence commands from `src/defenses` exactly as shown.
- **Dependencies/GPU:** Re-run `~/.local/bin/uv sync --locked --python 3.12`; check TensorFlow GPU discovery above and verify Docker's NVIDIA runtime separately. CPU execution is possible for local scripts but slow.
- **Out of memory:** Reduce `batch` in training/evaluation YAML, `data.batch_size` in the temporary PGD YAML, or Pixle `--workers`; record any changed setting. Randomised smoothing and full combination tests can be expensive.
- **Exact historical reproduction:** The two-source merge/deduplication procedure and per-model historical training YAML snapshots are absent. The checked-in curated split supports the executable pipeline, but raw-to-curated reconstruction and byte-identical historical weights cannot be verified from this repository alone.
