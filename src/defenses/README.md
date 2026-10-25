# Phase 3: Defences

Defences against the Phase 2 attacks (white-box PGD, black-box Pixle) for the Phase 1 ResNet50 classifier. No defence is trained or tuned on attacked images, and no defence setting is taken from an attack setting. The full plan and the leakage rules are in [PRD_defences.md](PRD_defences.md).

## Setup

- Base classifier: `best_robust.keras` at the repo root (set as `phase1_model` in `config.yaml`, gitignored), or pass `--model`.
- Data: `data/data_updated/` at the repo root, or pass `--data`. Its `data.yaml` is the source of truth for the class order and for which folders are the `train`/`val`/`test` splits (read by `config_loader.py`). To tune a defence against an attack, point `val:` in `data.yaml` at an attacked-val folder and re-run; the test splits stay untouched.
- Build the image once, from `src/defenses/`:

```bash
cd src/defenses
docker compose build
```

## Commands

Run all commands from `src/defenses/`.

### Test the preprocessing-based defences

```bash
# all 16 pipelines on the three test sets
docker compose run --rm -e CUDA_VISIBLE_DEVICES=-1 phase3 python modules/preprocessing/test_defences.py

# Eg. only the pipelines made of some defences (combinations can only contain jpeg, median) 
docker compose run --rm -e CUDA_VISIBLE_DEVICES=-1 phase3 python modules/preprocessing/test_defences.py --defence jpeg median

# predict everything again, ignoring saved results
docker compose run --rm -e CUDA_VISIBLE_DEVICES=-1 phase3 python modules/preprocessing/test_defences.py --rerun
```

- `-e CUDA_VISIBLE_DEVICES=-1` runs on CPU. The image uses TF 2.16.2, which cannot run its GPU kernels on an RTX 50-series card (`CUDA_ERROR_INVALID_PTX`). Leave the flag out on a supported GPU.
- A fresh run takes about 45 minutes on CPU. A pairing that already has `predictions.csv` is reused, so a repeat run takes seconds.
- With `--defence`, `summary.csv` and `summary.md` list only that run's pipelines. A plain run restores the full table.
- The `## Analysis` section of `summary.md` is written by hand and is kept when the table is regenerated.

### Tune the preprocessing-based defences

```bash
# sweep each defence's grid on the val split (from data.yaml), freeze the best val macro-F1 setting
docker compose run --rm -e CUDA_VISIBLE_DEVICES=-1 phase3 python modules/preprocessing/tune_preprocessing.py
```

Writes `modules/preprocessing/tuned_defences.yaml` and `tuning_logs/preprocessing/<defence>/grid.csv`. Afterwards `test_defences.py` uses the tuned settings automatically (it falls back to the fixed defaults when the file is absent, and prints which source it used).

### Tune and test the post-processing defences

```bash
# tune on val, then test alone + all ordered chains on the clean and attacked test sets
docker compose run --rm phase3 bash modules/post-processing/run_all.sh
```

Three ART output-side defences (Gaussian noise, high-confidence thresholding, reverse sigmoid). See [modules/post-processing/README.md](modules/post-processing/README.md) for the defences, the tuning objective, and why `Rounded` is excluded.

### Train and test the augmentation-based defences

```bash
# train both grids on train/, choose on clean val/, write the loss report
docker compose run --rm phase3 bash modules/augmentation-based/run_all.sh

# test the chosen models against the base classifier
docker compose run --rm phase3 python modules/augmentation-based/eval_augmentation.py
```

- Training needs a GPU that the image supports.
- `eval_augmentation.py` stops with "Model not found" until the models are retrained.
- Quick smoke test of one training run: `python modules/augmentation-based/mixup.py --alpha 0.2 --epochs 2`.

### Without Docker

Use Python 3.9 to 3.12 (TF 2.16 does not support newer versions), then:

```bash
pip install tensorflow==2.16.2 -r requirements.txt
python modules/preprocessing/test_defences.py
```

## What has been tested so far

| Defence type | Defences | Status |
|---|---|---|
| Preprocessing-based | JPEG compression, median smoothing, Gaussian randomized smoothing | **Tested**, alone and chained. Settings can now be tuned on val (`tune_preprocessing.py`) |
| Post-processing-based | Gaussian noise, high-confidence thresholding, reverse sigmoid (ART) | Scripts added: tuned on val, tested alone and chained. Output-side defences — see `modules/post-processing/` |
| Augmentation-based | Random erasing, mixup (retrained models) | **Not tested yet.** Needs retraining on the current data (see below) |

### Preprocessing-based defences

Each defence filters the image at test time, in front of the unchanged base classifier. The settings are fixed, commonly used defaults. They are not tuned.

| Defence | Fixed setting |
|---|---|
| JPEG compression | quality 75 |
| Median smoothing | 3 x 3 kernel |
| Gaussian randomized smoothing | σ = 8 (0-255 scale), 32 noisy copies, majority vote, seed 4266 |

The test covers 16 pipelines: no defence, each defence alone, and every ordered chain of two and three defences. Each pipeline is tested on three sets of 115 images from `data/data_updated/`:

- `test/`: clean images, to show what a defence costs on normal inputs
- `pgd_attacked/`: white-box attack
- `pixle_attacked/`: black-box attack

Accuracy of the baseline, the single defences and the best chains:

| Pipeline | Clean | PGD | Pixle |
|---|---|---|---|
| no defence | 100% | 0.0% | 60.0% |
| jpeg | 99.1% | 17.4% | 95.7% |
| median | 100% | 2.6% | 100% |
| gaussian | 98.3% | 3.5% | 96.5% |
| jpeg → gaussian (best pair) | 98.3% | 74.8% | 98.3% |
| jpeg → gaussian → median | 100% | 80.0% | 100% |
| jpeg → median → gaussian (best on PGD) | 98.3% | 87.0% | 98.3% |

- Clean accuracy stays at 98.3% or higher for every pipeline.
- Any defence handles Pixle (95.7% or higher).
- PGD needs chained defences: single defences reach at most 17.4%, three-defence chains 73% to 87%.
- The PGD images were crafted against the undefended model, so the PGD figures are an upper bound.

All 16 pipelines and the written analysis are in [testing_results/preprocessing_based/summary.md](testing_results/preprocessing_based/summary.md).

The metrics are the ones Phase 2 used. The test script imports the metric and report functions of `src/modelling/test.py` and calls them unchanged, and its no-defence `classification_report.txt` is identical to Phase 2's on all three sets.

### Augmentation-based defences

Random erasing and mixup are training-time defences: each is a separately retrained ResNet50.

- An earlier round trained and tuned them on the old split (`data/data_processed/`: 865 train, 257 clean val) and chose 0.1 for both. The record is in `tuning_logs/augmentation_based/`.
- The base classifier now uses `data/data_updated/`, so those models no longer match it. Their checkpoints were deleted and they have not been tested on the current attacked sets.
- Next step: retrain both on `data/data_updated/train/` and tune on clean `val/` only (PRD section 5).

## Outputs

```
testing_results/preprocessing_based/
  summary.md                         ranked table of all 16 pipelines + written analysis
  summary.csv                        one row per pipeline and test set
  <pipeline>/<test|pgd_attacked|pixle_attacked>/
    classification_report.txt        per-class precision, recall, f1-score, support
    test_metrics.txt                 all overall and per-class metrics
    confusion_matrix.txt
    confusion_matrix_normalized.txt
    predictions.csv                  every image: true class, predicted class, probabilities
testing_results/post_processing_based/  same five files per chain/test set + summary.{csv,md} (accuracy, macro F1, and clean-split obfuscation)
testing_results/augmentation_based/  eval_augmentation.py: results.csv, summary.csv (empty until retraining)
tuning_logs/preprocessing/           per-setting val scores from tune_preprocessing.py
tuning_logs/post-processing/         per-setting val scores + obfuscation from tune_postprocessing.py
tuning_logs/augmentation_based/      record of the earlier tuning on the old split
checkpoints/augmentation/            trained models and per-run histories (gitignored)
modules/preprocessing/tuned_defences.yaml        frozen settings of the preprocessing defences (if tuned)
modules/post-processing/tuned_defences.yaml      frozen settings of the post-processing defences
modules/augmentation-based/tuned_defences.yaml   frozen settings of the augmentation-based defences
```

Pipeline folders are named `no_defence`, `jpeg`, `jpeg_then_median`, `jpeg_then_median_then_gaussian`, and so on. Chains apply left to right.

For any chain with Gaussian smoothing, the class scores are vote shares over the 32 noisy copies, not model probabilities. Compare accuracy and F1 across pipelines, not loss or ROC AUC.

## Code

| File | Purpose |
|---|---|
| `modules/preprocessing/defences.py` | The three preprocessing defences and the chain predictor |
| `modules/preprocessing/tune_preprocessing.py` | Tunes the preprocessing defences on val, freezes the winners |
| `modules/preprocessing/test_defences.py` | Tests the preprocessing pipelines and writes the reports |
| `modules/post-processing/defences.py` | The three ART post-processing defences and the chain predictor |
| `modules/post-processing/tune_postprocessing.py` | Tunes the post-processing defences on val (max obfuscation within tolerance) |
| `modules/post-processing/test_postprocessing.py` | Tests the post-processing pipelines, alone and chained |
| `modules/augmentation-based/random_erasing.py`, `mixup.py` | Train one model per grid value |
| `modules/augmentation-based/_trainer.py` | Shared training loop (the Phase 1 recipe) |
| `modules/augmentation-based/report.py` | Builds the loss report from the training histories |
| `modules/augmentation-based/eval_augmentation.py` | Tests the retrained models |
| `modules/evaluation.py`, `modules/config_loader.py` | Shared command line, config and model loading |
| `config.yaml` | Paths and Phase 3 settings; everything else comes from `../modelling/configs/train.yaml` |
