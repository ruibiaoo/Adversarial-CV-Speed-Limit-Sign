# PRD: Phase 3 leakage-free defences

Last updated: 2026-10-14. Owner: Phase 3.

## 1. Goal and status

Make the Phase 1 classifier more robust to the Phase 2 attacks, without using attack data or attack knowledge in any defence. The test scores then measure robustness to an attack the defence has never seen.

| Defence type | What it is | Status |
|---|---|---|
| Preprocessing-based (section 4) | Filters applied to the image at test time, in front of the unchanged base classifier | **Tested** with fixed settings, alone and chained. Not tuned. |
| Augmentation-based (section 5) | New ResNet50 models retrained with an extra training-time augmentation | **To be retrained and tuned on clean val.** The earlier models were trained on the old data split and have been deleted. |

Base classifier: `src/modelling/runs/tf_resnet50_speed_limit_aug/best/best.keras` (TF ResNet50, 256 px, trained on `data/data_updated/train/`). The file is gitignored.

## 2. Data

All splits are in `data/data_updated/`.

| Split | Images | Used for |
|---|---|---|
| `train/` | 1080 | Training the augmentation-based models |
| `val/` (clean) | 220 | Tuning: checkpoint choice and hyperparameter choice |
| `test/` (clean) | 115 | Final testing only |
| `pgd_attacked/` | 115 | Final testing only. White-box PGD on the 115 test images |
| `pixle_attacked/` | 115 | Final testing only. Black-box Pixle on the same 115 images |

The base classifier scores 100% on `test/`, so both attacked sets cover every test image.

## 3. The no-leakage rule

A defence leaks if its training data, tuning data, settings or selection rule depends on the Phase 2 attacks.

- **Data leakage:** any attacked image is used for training, tuning, early stopping or checkpoint choice. This covers the Phase 2 test sets, attacked copies of `train/` or `val/`, and adversarial examples generated during training.
- **Knowledge leakage:** a defence setting is copied or derived from an attack setting. Examples: training against ε = 8/255 because PGD used it, or choosing a 3 × 3 median kernel because Pixle uses 3 × 3 patches.

| Data | Train a defence | Tune / choose settings | Final testing |
|---|---|---|---|
| `train/` (clean) | yes (augmentation-based only) | no | no |
| `val/` (clean) | no | **yes, the only tuning data** | no |
| `test/` (clean) | no | no | yes |
| `pgd_attacked/`, `pixle_attacked/` | **no** | **no** | yes |
| Attacked copies of train or val (any attack) | **no** | **no** | no |
| Attack settings (ε, step size, patch size, queries) | **no** | **no** | n/a |

Only two scripts read the test folders: `modules/preprocessing/test_defences.py` and `modules/augmentation-based/eval_augmentation.py`.

## 4. Preprocessing-based defences

### 4.1 Methods

Each defence changes the input image at test time. The base classifier is not retrained. Code: `modules/preprocessing/defences.py`.

| Defence | Method | Fixed setting | Why it is attack-agnostic |
|---|---|---|---|
| JPEG compression | Encode and decode the image as JPEG | quality 75 | Generic removal of high-frequency detail (Das et al. 2017; Guo et al. 2018). 75 is the common default quality. |
| Median smoothing | k × k median filter per channel | 3 × 3 | Standard denoiser for impulse and local corruption. 3 × 3 is the smallest kernel. |
| Gaussian randomized smoothing | 32 copies with N(0, σ²) noise; the prediction is the majority vote (Cohen et al. 2019) | σ = 8 on the 0-255 scale, seed 4266 | Averages the model over random noise, so the decision depends less on any small input change. No certified radius is claimed. |

### 4.2 Settings are fixed, not tuned

The settings above are commonly used defaults, chosen before any test run and not taken from any attack setting. No tuning was done: the planned tuning script (`tune_preprocessing.py`) was never run and has been removed, along with `eval_preprocessing.py`. This has zero leakage, but the settings are not optimised.

### 4.3 What was tested

`modules/preprocessing/test_defences.py` tests 15 pipelines on the three test sets:

- each defence alone (3)
- every ordered chain of two defences (6) and of three defences (6); order matters

In a chain, the filters run left to right. Gaussian smoothing makes its 32 noisy copies at its position, the later filters run on every copy, and the predictions are majority-voted.

Metrics are the ones Phase 2 used: the script imports the metric and report functions of `src/modelling/test.py` and calls them unchanged. The undefended classifier is not tested by this script, because Phase 2 already tested it with `src/modelling/test.py`. An earlier run that included it produced a `classification_report.txt` identical to Phase 2's on all three test sets.

### 4.4 Results

Accuracy on 115 images per test set. One image is about 0.87 points.

| Pipeline | Clean | PGD | Pixle |
|---|---|---|---|
| no defence (Phase 2 result) | 100% | 0.0% | 60.0% |
| jpeg | 99.1% | 17.4% | 95.7% |
| gaussian | 98.3% | 3.5% | 96.5% |
| median | 100% | 2.6% | 100% |
| jpeg → gaussian | 98.3% | 74.8% | 98.3% |
| median → gaussian | 99.1% | 53.0% | 99.1% |
| gaussian → jpeg | 98.3% | 49.6% | 97.4% |
| gaussian → median | 100% | 49.6% | 100% |
| jpeg → median | 100% | 48.7% | 100% |
| median → jpeg | 99.1% | 46.1% | 98.3% |
| jpeg → median → gaussian | 98.3% | **87.0%** | 98.3% |
| median → jpeg → gaussian | 98.3% | 82.6% | 99.1% |
| jpeg → gaussian → median | **100%** | 80.0% | **100%** |
| gaussian → median → jpeg | 99.1% | 76.5% | 100% |
| median → gaussian → jpeg | 99.1% | 73.0% | 99.1% |
| gaussian → jpeg → median | 100% | 73.0% | 100% |

Findings:

- **Clean cost is small.** Every pipeline scores 98.3% to 100% on clean images. Only two test images are ever misclassified.
- **Pixle is handled by any defence.** Accuracy rises from 60% to 95.7% or more.
- **PGD needs stacking.** Single defences reach at most 17.4%. Pairs average 53.6% and triples 78.7%.
- **Order matters.** JPEG works best first and Gaussian smoothing best last: jpeg → gaussian scores 74.8% on PGD, the reverse order 49.6%.

The full analysis is in `testing_results/preprocessing_based/summary.md`.

### 4.5 Outputs

```
testing_results/preprocessing_based/
  summary.md                       ranked table and written analysis
  summary.csv                      one row per pipeline and test set
  <pipeline>/<test|pgd_attacked|pixle_attacked>/
    classification_report.txt      per-class precision, recall, f1-score, support
    test_metrics.txt               all overall and per-class metrics
    confusion_matrix.txt
    confusion_matrix_normalized.txt
    predictions.csv
```

Pipeline folders are named `jpeg`, `jpeg_then_median`, `jpeg_then_median_then_gaussian`, and so on.

### 4.6 Limitations

- **The PGD attack did not know about the defences.** The PGD images were crafted against the undefended model. An adaptive attack on the full pipeline (BPDA, EOT; Athalye et al. 2018) would likely do better, so the PGD figures are an upper bound.
- **The ranking was made on the test set.** With 15 pipelines compared on 115 images, the best pipeline's score is a selected score, not an unbiased estimate.
- **Small sample.** Differences of 1 to 3 images are not meaningful on their own.
- **Vote shares are not probabilities.** Any chain with Gaussian smoothing outputs vote shares, so loss, ROC AUC and average precision are not comparable with the other pipelines. Compare accuracy and F1.
- **Randomness.** Gaussian smoothing depends on the noise draw. The seed is fixed, but another seed can move accuracy by about one image.

## 5. Augmentation-based defences

### 5.1 Methods

Each defence is a new ResNet50 trained with the Phase 1 recipe plus one extra per-batch augmentation. Code: `modules/augmentation-based/`, sharing `_trainer.py`.

| Defence | Method | Grid | Why it is attack-agnostic |
|---|---|---|---|
| Random erasing (Zhong et al. 2020) | With p = 0.5, one random rectangle (area 0.02 to max_area, aspect 0.3 to 3.3) is replaced with uniform noise | max_area ∈ {0.1, 0.2, 0.3} | Teaches the model not to rely on any single region of the sign. Models real occlusion such as stickers and dirt. |
| Mixup (Zhang et al. 2018) | Blend image pairs and their labels, λ ~ Beta(α, α) | α ∈ {0.1, 0.2, 0.4} | Keeps the decision boundary further from each training image, so small input changes of any kind are less likely to flip the class. |

The grids are the ranges of the original papers. Nothing in them comes from an attack setting.

### 5.2 Status

- An earlier round trained and tuned both defences on the **old** split: `data/data_processed/train/` (865 images) and clean `data/data_processed/val/` (257 images). It chose 0.1 for both. The record is in `tuning_logs/augmentation_based/`.
- The base classifier now uses `data/data_updated/`, so those models no longer match it. Their checkpoints were deleted and they have not been tested on the new attacked sets.
- `modules/augmentation-based/tuned_defences.yaml` still holds the old choices and points at the deleted checkpoints. The retraining in 5.3 overwrites it.

### 5.3 Tuning plan: clean val only

1. **Train.** For each grid value, train one model on `data/data_updated/train/` (1080 images) with the Phase 1 recipe: ResNet50, 256 px, the Phase 1 augmentations, Adam at 3e-4, batch 16, seed 4266, learning-rate reduction and early stopping on val macro F1.
2. **Choose the checkpoint.** For each run, keep the epoch with the best macro F1 on clean `data/data_updated/val/` (220 images).
3. **Choose the setting.** For each defence, pick the grid value with the best clean-val macro F1. Ties go to the milder setting (the smaller value).
4. **Freeze.** Write the chosen setting and checkpoint to `modules/augmentation-based/tuned_defences.yaml`. Testing reads settings only from this file.

Why clean val is the right tuning data:

- It is attack-agnostic: no attacked image and no attack setting is involved, so section 3 holds.
- It selects the model that keeps the most clean accuracy. It does not show how much robustness a setting buys. The robustness argument comes from the method itself, not from the tuning.
- The consequence is deliberate: the chosen setting may be weaker than the "best against PGD or Pixle" setting, but the test scores are honest.

Tuning on attacked images, or on other attacks such as FGSM, is not allowed for selection (section 3).

### 5.4 Testing plan

After tuning, `eval_augmentation.py` tests the chosen random-erasing and mixup models, with the base classifier as the no-defence baseline, on `test/`, `pgd_attacked/` and `pixle_attacked/`. It stops if a defence has no frozen setting or its checkpoint is missing.

It currently reports accuracy with a 95% Wilson interval, macro F1 and an exact McNemar p-value against no defence, in `testing_results/augmentation_based/results.csv` and `summary.csv`.

One point to interpret with care: the PGD and Pixle images were crafted against the base classifier. For a retrained model they are transferred attacks, which are usually weaker than attacks crafted against that model directly.

### 5.5 Open items

- [ ] Retrain both grids on `data/data_updated/` and freeze the choices (5.3). Training needs a GPU; TF 2.16.2 does not run on an RTX 50-series card (see section 8).
- [ ] `eval_augmentation.py` was moved into `modules/augmentation-based/`. Confirm it runs from there (it imports `modules` from `src/defenses/`).
- [ ] The training code writes its tuning record to `tuning_logs/augmentation/`, while the existing record is in `tuning_logs/augmentation_based/`. Make the two names agree before retraining.
- [ ] Make `eval_augmentation.py` write the same five Phase 2 report files as the preprocessing test, so both defence types are compared on the same metrics.
- [ ] Decide whether to also test the preprocessing pipelines in front of the retrained models.

## 6. Excluded: TRADES and all adversarial training

TRADES was removed from Phase 3, for two reasons:

- **It generates adversarial data:** PGD examples are made from `train/` during training.
- **Its budget came from the attack:** ε = 8/255 and step 2/255 were copied from the Phase 2 PGD.

That is both data leakage and knowledge leakage under section 3. PGD-AT and Fast-AT are excluded for the same reason.

## 7. Acceptance criteria

Preprocessing-based:

- [x] No training or tuning script reads an attacked folder.
- [x] No defence setting is taken from an attack setting.
- [x] All 15 pipelines are tested on the clean, PGD and Pixle sets.
- [x] The metrics match Phase 2: an earlier run with the undefended classifier reproduced Phase 2's `classification_report.txt` on all three sets.
- [x] Results and analysis are in `testing_results/preprocessing_based/`.

Augmentation-based:

- [ ] Both grids are retrained on `data/data_updated/train/`, with checkpoint and setting chosen on clean `val/` only.
- [ ] `modules/augmentation-based/tuned_defences.yaml` holds the new choices.
- [ ] The chosen models are tested on the clean, PGD and Pixle sets, with results in `testing_results/augmentation_based/`.

## 8. How to run

From `src/defenses/`:

```bash
docker compose build

# preprocessing-based: test all 15 pipelines (finished pairings are reused; --rerun forces them)
docker compose run --rm -e CUDA_VISIBLE_DEVICES=-1 phase3 python modules/preprocessing/test_defences.py

# augmentation-based: train both grids and freeze the choices, then test
docker compose run --rm phase3 bash modules/augmentation-based/run_all.sh
docker compose run --rm phase3 python modules/augmentation-based/eval_augmentation.py
```

`-e CUDA_VISIBLE_DEVICES=-1` runs on CPU. The image uses TF 2.16.2, which cannot run its GPU kernels on an RTX 50-series card (`CUDA_ERROR_INVALID_PTX`). A fresh preprocessing test takes about 45 minutes on CPU.
