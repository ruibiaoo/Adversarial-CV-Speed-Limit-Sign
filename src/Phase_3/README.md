# Phase 3: Defences

Fixes and defences against the Phase 2 attacks. The goal is to make the Phase 1 model robust to them, with **no data leakage** (no training on Phase 2 adversarial examples).

## Defences

One augmentation-based defence per subcategory, in `modules/augmentation-based/`. The choices are explained in the section below.

| Defence | Type | Command |
|---|---|---|
| Median smoothing | input transformation (test time) | `modules/augmentation-based/median_smoothing.py` |
| Random erasing | noise / corruption augmentation (training) | `modules/augmentation-based/random_erasing.py` |
| Mixup | label / mixing regularisation (training) | `modules/augmentation-based/mixup.py` |
| TRADES | adversarial augmentation (training; PGD made on the fly from `train/`, never saved) | `modules/augmentation-based/trades.py` |

Defence parameters (PGD budget ε / step / steps, ...) are in `config.yaml` and the scripts' grids. Everything else (model, optimizer, augmentation, seed) comes from `../Phase_1/config.yaml`.

## Which data is used for what

Splits come from `data/data_processed/` (865 train / 257 val / 131 test). Augmented copies of the same sign stay in the same split. All phases use the same splits.

| Data | Defence training | Tuning / checkpoint choice | Final evaluation |
|---|---|---|---|
| `train/` | yes | no | no |
| `val/` | no | yes (clean images only) | no |
| `test/` clean | no | no | yes |
| Phase 2 attacked images (made from `test/`) | **no** | **no** | yes |

- The scripts in `modules/augmentation-based/` read only `train/` and `val/`.
- `eval_defence.py` reads `test/`, or a folder passed with `--data-dir`.
- Median smoothing trains nothing. Its kernel is chosen on clean `val/` only.

## Run

```bash
cd Phase_3
docker compose build

# train / calibrate all four defences, then write the loss report (resumes after a crash)
docker compose run --rm phase3 bash modules/augmentation-based/run_all.sh
```

Notes:

- In Docker the repo is mounted at `/repo`, so `--data-dir` paths start with `/repo/`.
- Quick smoke test: add `--epochs 2` to any training script, e.g. `python modules/augmentation-based/mixup.py --alpha 0.2 --epochs 2`.
- Without Docker: `pip install -r requirements.txt`, then run the same `python ...` commands without the `docker compose run --rm phase3` prefix, from inside `Phase_3/`.
- The Phase 1 model is expected at `Phase_1/runs/tf_resnet50_speed_limit_aug/best/best.keras` (gitignored). Use `--model` if it is elsewhere.
- `eval_defence.py` still imports `modules/input_transform/`, which no longer exists, so it does not run yet. It needs updating for the defences above before the test-set evaluation.

## Chosen augmentation-based defences (one per subcategory)

Code is in `modules/augmentation-based/`, one script per defence. Each script explains its choice in its docstring. The folder name has a hyphen, so run the scripts directly instead of importing them. They share `_trainer.py`, which holds the data pipeline and the Phase 1 training recipe.

| # | Subcategory | Chosen | Script |
|---|---|---|---|
| 1 | Input transformation | Median smoothing | `median_smoothing.py` |
| 2 | Noise / corruption augmentation | Random erasing | `random_erasing.py` |
| 3 | Label / mixing regularisation | Mixup | `mixup.py` |
| 4 | Adversarial augmentation | TRADES | `trades.py` |

The attacks to defend against are PGD (white-box, L∞ ε = 8/255) and Pixle (black-box, overwrites 3x3 pixel patches with pixels copied from elsewhere in the image, about 560 pixels per image). Each defence is picked for how well it fits these attacks and the selection rule min(clean, WB, BB), where clean accuracy counts as much as robustness.

### 1. Median smoothing (input transformation)

**Method.** At test time, each image is median filtered with a k x k window before it goes to the frozen model: each pixel becomes the median of its neighbourhood. No retraining.

**Why it was chosen.**
- It directly targets the black-box attack. A median filter removes any outlier region that covers less than half the window. A 3x3 Pixle patch fills 9 of 25 pixels in a 5x5 window, so k ≥ 5 removes it completely. A check on a synthetic patch confirmed this: k = 3 leaves 5 pixels, k = 5 leaves none.
- It also smooths the high-frequency PGD noise, so it covers both attacks. JPEG, the usual alternative, only helps against PGD.
- It keeps edges, so the large, high-contrast digits survive and clean accuracy stays close to no defence. Gaussian blur smears the edges.
- It is cheap, has a single parameter (k), and works with any model.

### 2. Random erasing (noise / corruption augmentation)

**Method.** During training, with probability 0.5, one random rectangle in each image is replaced with random pixel values. The rectangle covers 2% up to a maximum area of the image, with aspect ratio 0.3 to 3.3. It is applied after the Phase 1 augmentations (Zhong et al., 2020).

**Why it was chosen.**
- Pixle's damage is local regions overwritten with unrelated content. Random erasing trains the model on exactly that kind of corruption, so it learns to classify from what remains.
- It is also the physical-world version of the threat: stickers, dirt, graffiti and partial occlusion on real signs.
- The existing Gaussian augmentation only covers dense, small noise, which is a weak defence against PGD at 8/255. Random erasing adds the sparse, large-amplitude case that Gaussian noise misses.
- The maximum area is capped at 0.3, because a larger rectangle can hide the whole digit and make the label wrong.

### 3. Mixup (label / mixing regularisation)

**Method.** Each training image is blended with another image from the batch: x = λ·x_i + (1 − λ)·x_j, with λ ~ Beta(α, α). The one-hot labels and class weights are blended with the same λ (Zhang et al., 2018).

**Why it was chosen.**
- Mixup makes the model behave close to linearly between training images. This keeps the decision boundary away from each image and reduces over-confidence. The original paper reports better robustness to FGSM / I-FGSM, which are PGD-type attacks.
- Label smoothing, the existing method here, mostly makes PGD gradients less useful (gradient masking, PRD §10). That does not hold up against a stronger attacker, so it is not real robustness.
- CutMix pastes a rectangle from another sign and weights the label by area. The class depends only on the digits, so if the patch covers the digits the label is wrong. Mixup blends the whole image, so both signs' digits stay visible in proportion to their label weights.
- It adds no training cost.

### 4. TRADES (adversarial augmentation)

**Method.** Each training batch gets a fresh adversarial copy x_adv: 5 PGD steps inside the L∞ ball ε = 8/255 that maximise KL(f(x) ‖ f(x_adv)). The model is trained on CE(f(x), y) + β·KL(f(x) ‖ f(x_adv)) (Zhang et al., 2019). The adversarial copies are made from the train split only, kept in memory and never saved (PRD §3).

**Why it was chosen.**
- Adversarial training is the only subcategory with a strong record against adaptive attacks, and ε = 8/255 matches the Phase 2 PGD budget.
- TRADES keeps the clean term and the robustness term separate, and β trades one against the other. The selection rule takes the worst of clean, WB and BB, so this explicit control over clean accuracy matters. Plain PGD adversarial training only has a coarse clean/adversarial batch ratio.
- The KL term pulls f(x_adv) towards f(x), not towards a hard label. This gives smoother decision boundaries, which also helps against an attack it was not trained on (Pixle).
- Fast-AT only saves compute. With 865 train images that is not needed, and single-step training risks catastrophic overfitting.

### Results on clean val

| Defence | Chosen setting | Val accuracy | Val macro F1 | Val loss |
|---|---|---|---|---|
| Median smoothing | not calibrated yet (needs the Phase 1 checkpoint) | - | - | - |
| Random erasing | max area = 0.1 | 0.9961 | 0.9980 | 0.021 |
| Mixup | α = 0.1 | 1.0000 | 1.0000 | 0.030 |
| TRADES | β = 1 | 0.9883 | 0.9854 | 0.068 |

These are clean-val numbers, used only to choose parameters. The defences are compared on the WB/BB test sets later (PRD §4, §6). Choosing β on clean val will always favour the smaller β, so evaluate both TRADES checkpoints (β = 1 and β = 6) on the attacked sets. The settings within each defence are close (e.g. random erasing max area 0.3 scored 0.991, Mixup α = 0.4 scored 0.996), and the val curves swing a lot from epoch to epoch, so these picks are near ties and change between reruns. Median smoothing is calibrated on the Phase 1 checkpoint, so it needs that file on the machine you run on.

### Training and validation loss (`tuning/augmentation-tuning/`)

The training scripts write each run's per-epoch history here. `report.py` then builds the loss report from them. `run_all.sh` runs it last.

```
tuning/augmentation-tuning/
  loss_report.md              # start here: loss curves + per-run table for each defence
  summary.csv                 # one row per run: best epoch, train / val loss, val acc / macro F1, chosen
  random_erasing_loss.png     # train loss and val loss per epoch, one line per setting,
  mixup_loss.png              #   (dot = checkpoint kept)
  trades_loss.png
  median_smoothing/grid.csv   # val loss / acc / macro F1 per kernel (no training, so no epochs)
  random_erasing/             # grid.csv + max_area<v>_history.csv
  mixup/                      # grid.csv + alpha<v>_history.csv
  trades/                     # grid.csv + beta<v>_history.csv
```

Each `*_history.csv` has one row per epoch: `epoch, train_loss, val_loss, val_accuracy, val_macro_f1, lr, seconds`.

- `train_loss` is the defence's own objective (on erased / mixed / adversarial inputs, plus TRADES' KL term). It is only comparable between settings of the same defence.
- `val_loss` is plain cross entropy on clean val, comparable across every run.
- To rebuild the report from existing histories: `python modules/augmentation-based/report.py`.

Parameters are chosen on clean `val/` only and frozen in `defence_configs.yaml`:

| Defence | Grid | Rule |
|---|---|---|
| Median smoothing | k ∈ {3, 5, 7} | largest k with clean-val accuracy drop ≤ 1 pt (tolerance fixed before running) |
| Random erasing | max area ∈ {0.1, 0.2, 0.3}, p = 0.5 | best clean-val macro F1 |
| Mixup | α ∈ {0.1, 0.2, 0.4} | best clean-val macro F1 |
| TRADES | β ∈ {1, 6}. ε = 8/255, step 2/255, 5 steps fixed a priori (Phase 2 PGD budget) | best clean-val macro F1 |

```bash
docker compose run --rm phase3 bash modules/augmentation-based/run_all.sh   # all four, in order
docker compose run --rm phase3 python modules/augmentation-based/mixup.py --alpha 0.2 --epochs 2   # one quick run
```

`run_all.sh` trains the three training defences, then calibrates median smoothing on the Phase 1 checkpoint (skipped with a warning if that file is missing; run `median_smoothing.py --model <path>` later), then writes the report. Checkpoints go to `runs/<defence>_<param><value>/best.keras`, per-epoch logs to `history.csv`, and grid results to `runs/<defence>_grid.csv`.

The first run on an RTX 50-series GPU spends about 30 min JIT-compiling kernels, because TF 2.15 has no native kernels for these GPUs. `docker-compose.yml` keeps the cache in `.cuda_cache/` so later runs skip this.
