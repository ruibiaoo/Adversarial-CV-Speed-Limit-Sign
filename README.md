# Defending Speed-Limit Sign Classification Against Adversarial Attacks

## Project Overview

This project uses a ResNet50 speed-limit sign classifier to evaluate PGD and Pixle attacks. It trains and compares input-level, training-stage and output-level defences, then generates Grad-CAM and Ablation-CAM visualisations.

## Getting Started

### Prerequisites

The project is developed for **Ubuntu/Linux** and requires:

- Python 3.12
- [uv](https://docs.astral.sh/uv/) — Python package manager
- Docker Engine and Docker Compose — for model training and evaluation
- NVIDIA GPU, drivers and NVIDIA Container Toolkit — for GPU-enabled Docker execution

Dependencies are managed using `pyproject.toml` and `uv.lock`, including TensorFlow 2.16.2 and `tf-keras` 2.16.0.

Local attack scripts can run without a GPU, although execution may be slower.

### Repository Setup

**1. Clone the repository**

```bash
git clone https://github.com/ruibiaoo/Adversarial-CV-Speed-Limit-Sign.git
cd Adversarial-CV-Speed-Limit-Sign
```

**2. Install uv**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**3. Set up the Python environment and install dependencies**

```bash
~/.local/bin/uv sync --locked --python 3.12
```

**4. Verify TensorFlow GPU availability (optional)**

```bash
~/.local/bin/uv run --locked python -c 'import tensorflow as tf; print(tf.config.list_physical_devices("GPU"))'
```

An empty list `[]` indicates that TensorFlow did not detect a GPU in the local Python environment.
## Dataset

### Data Sources

The speed-limit sign dataset was compiled from two publicly available sources:

1. **Kaggle — Speed Limit Dataset USA**  
   https://www.kaggle.com/datasets/basiradaqiq/speed-limit-dataset-usa

2. **Hugging Face — Speed Limit Signs**  
   https://huggingface.co/datasets/ctacke/speed-limit-signs

The final dataset contains **1,415 images** across seven speed-limit classes: 10, 20, 30, 40, 50, 60 and 70 MPH.

### Dataset Structure

The prepared dataset is available in `data/data_updated/` and is organised as follows:

| Split | Number of Images |
|---|---:|
| Training | 1,080 |
| Validation | 220 |
| Testing | 115 |
| **Total** | **1,415** |

```text
data/data_updated/
├── train/
│   ├── 10_MPH/
│   ├── 20_MPH/
│   └── ...
├── val/
│   ├── 10_MPH/
│   └── ...
├── test/
│   ├── 10_MPH/
│   └── ...
└── data.yaml
```

Images are loaded in RGB format and resized to 256 × 256 pixels for model input.


**The prepared dataset is already included in the repository.** You do not need to download the original datasets to run the experiments.

## Training-Stage Models
The Training-Stage Models can be found in this link: https://drive.google.com/drive/folders/1P13bIvg8HD2y2bCp0Rpw0hwUBuAz_M7c?usp=drive_link




## How to Run the Code

All commands should be executed from the **repository root**, unless otherwise specified. Complete the environment setup in [Getting Started](#getting-started) before proceeding.

### 1. Evaluate Existing Models (Quick Start)

If the models have already been trained, you can run the evaluation without retraining.

Ensure that the following checkpoints are available in `src/modelling/runs/`:

| Model | Checkpoint |
|---|---|
| M0 — Baseline ResNet50 | `best_baseline.keras` |
| M1 — Gradient Reversal Layer | `grl.keras` |
| M2 — Adversarial Retraining | `best_adversarial_retrained.keras` |
| M3 — Adversarial Objective Training | `best_robust.keras` |
| M4 — Fine-tuning with ConvAE | `best_convae_rn50.keras` |
| Denoising ConvAE | `convae_adv2clean_aug_v6.keras` |

**Note:** Model checkpoints are not included in a fresh clone. Obtain the pretrained models from the project authors or train them using the steps below.

Build the Docker image and run evaluation:

```bash
docker compose -f src/modelling/docker-compose.yml build

docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier python -u test_all.py
```

The evaluation covers clean, PGD, Pixle and combined adversarial images, together with the configured defence combinations.

Results are saved to:

`src/modelling/final_final_results_i_promise/`

If evaluating only selected models, update `src/modelling/configs/test_all.yaml` to enable the required models and defences.

---

### 2. Reproduce the Entire Experiment

Follow these steps in order to regenerate the adversarial datasets, train the models and evaluate the defences.

#### Step 1 — Train the Baseline ResNet50 (M0)

Open `src/modelling/configs/train.yaml` and configure:

| Parameter | Value |
|---|---|
| `data` | `/app/eval_data` |
| `project` | `/app/runs` |
| `pretrained` | `true` |
| `grl.enabled` | `false` |
| `convae.enabled` | `false` |
| `robust` | `false` |

Run:

```bash
docker compose -f src/modelling/docker-compose.yml run --build --rm phase1-classifier python -u train.py
```

Save the trained baseline:

```bash
cp src/modelling/runs/best.keras src/modelling/runs/best_baseline.keras
```

#### Step 2 — Generate Adversarial Datasets

Three attacked datasets are generated:

| Attack | Output Directory |
|---|---|
| PGD | `data/data_updated_pgd/` |
| Pixle | `data/data_updated_pixle/` |
| Pixle → PGD | `data/data_pgdpixle/` |

**PGD attack**

The existing PGD configuration targets Pixle images, so create a temporary configuration for clean images:

```bash
cp src/attack/pgd/configs/pgd.yaml /tmp/pgd_clean.yaml
sed -i 's#root: data/data_updated_pixle#root: data/data_updated#' /tmp/pgd_clean.yaml

~/.local/bin/uv run python src/attack/pgd/pgd_attacked_images_generation.py \
  --config /tmp/pgd_clean.yaml \
  --output-root data/data_updated_pgd \
  --splits train val test --overwrite
```

**Pixle attack**

```bash
~/.local/bin/uv run python src/attack/pixle/evaluate.py \
  --model src/modelling/runs/best_baseline.keras \
  --data data/data_updated \
  --splits train val test \
  --adversarial-dir data/data_updated_pixle \
  --results-dir src/attack/pixle/results --no-progress
```

**Combined attack (Pixle → PGD)**

Run this after generating the Pixle dataset:

```bash
~/.local/bin/uv run python src/attack/pgd/pgd_attacked_images_generation.py \
  --splits train val test --overwrite
```

**Note:** Commands using `--overwrite` may replace previously generated adversarial images.

#### Step 3 — Train the Denoising ConvAE

In `src/modelling/configs/train_convae.yaml`, set:

- `data: "/app/eval_data"`
- `adv_data: ["/app/data_pgdpixle"]`

Run:

```bash
docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier python -u train_convae.py
```

Copy the trained ConvAE checkpoint:

```bash
cp src/modelling/runs/convae/convae_adv2clean_aug_v6/best.keras \
  src/modelling/runs/convae_adv2clean_aug_v6.keras
```

#### Step 4 — Train the Defence Models (M1–M4)

First, combine the clean and adversarial training datasets:

```bash
~/.local/bin/uv run python src/data/combine_datasets.py
```

For each model, update `src/modelling/configs/train.yaml`:

- `data: "/app/data"`
- `project: "/app/runs"`
- `pretrained: "/app/runs/best_baseline.keras"`

Then configure the corresponding defence:

| Model | `grl.enabled` | `robust` | `convae.enabled` |
|---|---|---|---|
| M1 — GRL | `true` | `false` | `false` |
| M2 — Adversarial Retraining | `false` | `false` | `false` |
| M3 — Adversarial Objective Training | `false` | `true` | `false` |
| M4 — Fine-tuning with ConvAE | `false` | `false` | `true` |

For M4, ensure that `convae.model_path` points to the trained ConvAE checkpoint.

Run the following command **separately for each model**, after updating its configuration:

```bash
docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier python -u train.py
```

Save each trained checkpoint before starting the next model, since `best.keras` may be overwritten:

```bash
# After M2
cp src/modelling/runs/best.keras src/modelling/runs/best_adversarial_retrained.keras

# After M3
cp src/modelling/runs/best.keras src/modelling/runs/best_robust.keras

# After M4
cp src/modelling/runs/best.keras src/modelling/runs/best_convae_rn50.keras
```

M1 saves its GRL checkpoint separately as `grl.keras`.

#### Step 5 — Evaluate Defence Combinations

To run the standalone input-level and output-level defence experiments, first prepare the required combined evaluation dataset:

```bash
~/.local/bin/uv run python src/data/combine_clean_pgd_pixle.py
```

Then execute the defence scripts:

```bash
cd src/defenses

MODEL=../../src/modelling/runs/best_baseline.keras
DATA=../../data/data_clean_pgd_pixle

~/.local/bin/uv run python modules/preprocessing/tune_preprocessing.py --model "$MODEL" --data "$DATA"
~/.local/bin/uv run python modules/preprocessing/test_defences.py --model "$MODEL" --data "$DATA"

~/.local/bin/uv run python modules/post-processing/tune_postprocessing.py --model "$MODEL" --data "$DATA"
~/.local/bin/uv run python modules/post-processing/test_postprocessing.py --model "$MODEL" --data "$DATA"

cd ../..
```

For the full evaluation across the baseline, training-stage models and combined defence configurations, check `src/modelling/configs/test_all.yaml` and run:

```bash
docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier python -u test_all.py
```

The results will be saved to the output directory specified in `test_all.yaml`.

### 3. Generate Grad-CAM and Ablation-CAM Visualisations

The interpretability scripts are located at:

- `src/modelling/gradcam.py`
- `src/modelling/ablationcam.py`

Both require model checkpoints and matching `predictions.csv` files generated during evaluation.

The following example generates a Grad-CAM visualisation for a PGD-attacked 20 MPH sign using the baseline model:

```bash
docker compose -f src/modelling/docker-compose.yml run --rm phase1-classifier \
  python -u gradcam.py \
  --model /app/runs/best_baseline.keras \
  --predictions /app/final_final_results_i_promise/tests/pre_none__model_baseline__post_none/pgd_test/predictions.csv \
  --image '20s1_png.rf.21e5ce2e19a562face23463639c38966.png' \
  --image-size 256 \
  --target true \
  --output-dir /app/results/gradcam_20mph/baseline
```

To generate the corresponding Ablation-CAM, replace `gradcam.py` with `ablationcam.py` and change the output directory.

Use the matching evaluation CSV when generating heatmaps for other defence models or preprocessing combinations.


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

