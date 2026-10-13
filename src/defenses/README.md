# Phase 3: Defences

Defences against the Phase 2 attacks (white-box PGD, black-box Pixle) for the Phase 1 ResNet50 classifier, with **no data leakage and no attack knowledge**: no defence is trained or tuned on attacked images, and no defence setting is taken from an attack setting. The full plan, the leakage rules and the feasibility analysis are in [PRD_defences.md](PRD_defences.md).

## Defences

### Preprocessing defences (test time, same frozen Phase 1 model)

Code: `modules/preprocessing/defences.py`. Calibrated by `tune_preprocessing.py` on clean val.

| | Defence | Method | Grid (mildest first) |
|---|---|---|---|
| D1 | JPEG compression | encode and decode the input as JPEG at quality q | q ∈ {90, 75, 60, 50, 40, 30} |
| D2 | Gaussian randomized smoothing | 32 noisy copies, N(0, σ²), majority vote (seed 4266) | σ ∈ {4, 8, 12, 16, 24} (0-255 scale) |
| D3 | Median smoothing | k x k median filter | k ∈ {3, 5, 7} |

Rule, fixed before running: the strongest setting whose clean-val accuracy drop is at most 1 point.

### Augmentation-based defences (training time, retrained models)

Code: `modules/augmentation-based/`, sharing `_trainer.py` (the Phase 1 training recipe). The folder name has a hyphen, so run the scripts directly.

| Defence | Method | Grid | Chosen (clean val macro F1) |
|---|---|---|---|
| Random erasing | one random rectangle per image replaced with noise (p = 0.5) | max area ∈ {0.1, 0.2, 0.3} | 0.1 |
| Mixup | blend image pairs and labels, λ ~ Beta(α, α) | α ∈ {0.1, 0.2, 0.4} | 0.1 |

TRADES (adversarial training) was removed. It generated PGD examples during training, using the Phase 2 PGD budget, which is leakage under the PRD rules.

## Which data is used for what

Splits come from `data/data_processed/` (865 train / 257 val / 131 test). All phases use the same splits.

| Data | Defence training | Tuning / checkpoint choice | Final evaluation |
|---|---|---|---|
| `train/` | yes (augmentation defences) | no | no |
| `val/` | no | yes (clean images only) | no |
| `test/` clean | no | no | yes |
| `pgd_attacked/`, `pixle_attacked/` (made from `test/`) | **no** | **no** | yes |

- The training and tuning scripts read only `train/` and `val/`. `tune_preprocessing.py` refuses any other split.
- `eval_preprocessing.py` and `eval_augmentation.py` are the only scripts that read the test folders (`test_sets` in `config.yaml`). They stop if a requested defence has not been tuned yet.

## Run

```bash
cd Phase_3
docker compose build

# train the augmentation defences, calibrate the preprocessing defences, write the loss report
docker compose run --rm phase3 bash modules/augmentation-based/run_all.sh

# final evaluation (after tuning), one table per group
docker compose run --rm phase3 python eval_preprocessing.py --defence all   # 3 test sets x 4 = 12 pairings
docker compose run --rm phase3 python eval_augmentation.py --defence all    # 3 test sets x 3 = 9 pairings

# or test only some defences (no defence is always included as the baseline)
docker compose run --rm phase3 python eval_preprocessing.py --defence jpeg median
docker compose run --rm phase3 python eval_augmentation.py --defence mixup
```

`--defence` options: `jpeg`, `gaussian`, `median` (preprocessing); `random_erasing` (or `erasing`), `mixup` (augmentation); or `all` (the default). The full names in `tuned_defences.yaml` also work. Testing only some defences updates their rows and keeps the rest.

Both scripts report accuracy (95% Wilson CI), macro F1 and the McNemar p-value against no defence for each pairing. It picks the defence with the largest worst case, min(clean, WB, BB). Results go to `test_results/<group>/results.csv` and `summary.csv`.

Notes:

- The Phase 1 model is expected at `Phase_1/runs/tf_resnet50_speed_limit_aug/best/best.keras` (gitignored). Use `--model` if it is elsewhere, and `--data` if `data_processed` is not found.
- Quick smoke test of a training defence: `python modules/augmentation-based/mixup.py --alpha 0.2 --epochs 2`.
- Without Docker: `pip install -r requirements.txt`, then run the same `python ...` commands from inside `Phase_3/`.
- The first run on an RTX 50-series GPU spends about 30 min JIT-compiling kernels (TF 2.15 has no native kernels for these GPUs). `docker-compose.yml` keeps the cache in `.cuda_cache/`.

## Outputs

```
tuned_defences.yaml                             frozen settings of every defence (written by tuning / training)
checkpoints/augmentation/<defence>_<param><v>/   best.keras + history.csv per training run
checkpoints/augmentation/                        <defence>_grid.csv, logs/
checkpoints/preprocessing/                       <defence>_grid.csv per tuning run, logs/
tuning_logs/augmentation/                        loss_report.md, summary.csv, <defence>_loss.png, histories
tuning_logs/preprocessing/                       <defence>_grid.csv (clean-val accuracy / macro F1 / drop), summary.csv
test_results/<group>/                            results.csv, summary.csv (final evaluation)
```

- `checkpoints/` holds everything a run produces, including the models. It is gitignored because the checkpoints are too large for GitHub.
- `tuning_logs/` is the small, committed record of how each setting was chosen.
- Every defence group has its own folder under both (see PRD section 5.1). The scripts create them if they are missing.

Each `*_history.csv` has one row per epoch: `epoch, train_loss, val_loss, val_accuracy, val_macro_f1, lr, seconds`. `train_loss` is the defence's own objective. `val_loss` is plain cross entropy on clean val, so it is comparable across runs. Rebuild the report with `python modules/augmentation-based/report.py`.
