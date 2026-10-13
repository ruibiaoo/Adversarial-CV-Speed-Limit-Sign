# PRD: Phase 3 leakage-free defences

Status: planned and implemented in code; tuning and evaluation runs are pending (they need the Phase 1 checkpoint).
Owner: Phase 3. Model: Phase 1 TF ResNet50 (`Phase_1/runs/tf_resnet50_speed_limit_aug/best/best.keras`).

## 1. Goal

Make the Phase 1 classifier more robust to the Phase 2 attacks. Compare defences on three test sets and pick the one with the best worst case. **No defence may use attack data or attack knowledge.** The test-set scores then measure robustness to an attack the defence has never seen, rather than how well it fits that attack.

The three test sets:

| Condition | Folder | Images | Made by |
|---|---|---|---|
| Clean | `data/data_processed/test/` | 131 | Phase 1 split |
| White-box (WB) | `data/data_processed/pgd_attacked/` | 130 | Phase 2 PGD on the test images Phase 1 classified correctly |
| Black-box (BB) | `data/data_processed/pixle_attacked/` | 130 | Phase 2 Pixle on the same 130 images |

## 2. The no-leakage rule

A defence leaks if any of its training data, tuning data, design parameters or selection rule depends on the Phase 2 attacks. There are two kinds.

**Data leakage.** Any attacked image is used for training, tuning, early stopping or checkpoint choice. This covers:
- the Phase 2 test sets;
- attacked copies of `train/` or `val/` made with the Phase 2 attacks;
- adversarial examples generated during training.

A defence tuned this way learns the attack, so its WB/BB score measures fit to that attack rather than robustness.

**Knowledge leakage.** A defence setting is copied or derived from the attacks. Examples:
- training against ε = 8/255 because Phase 2 PGD used 8/255;
- choosing a median kernel because Pixle uses 3×3 patches;
- tuning on FGSM or CW examples, which still tells the defence the attack family.

For the black-box attack this also breaks the threat model: the defender is not supposed to know the attack.

### Data-usage matrix

| Data | Train a defence | Tune / choose settings | Final evaluation |
|---|---|---|---|
| `train/` (clean) | yes (augmentation defences only) | no | no |
| `val/` (clean) | no | **yes, the only tuning data** | no |
| `test/` (clean) | no | no | yes |
| `pgd_attacked/`, `pixle_attacked/` | **no** | **no** | yes |
| Attacked copies of train/val (any attack) | **no** | **no** | no |
| Attack settings (ε, step, patch size, queries) | **no** | **no** | n/a |

`tune_preprocessing.py` enforces this in code: `check_tuning_split()` rejects any split other than `val`, and any path containing `attacked`. The evaluation scripts `eval_preprocessing.py` and `eval_augmentation.py` are the only scripts that read the test folders. They refuse to test a defence that has no frozen setting in `tuned_defences.yaml`.

## 3. Feasibility of tuning on each candidate data source

| Candidate tuning data | Verdict | Reason |
|---|---|---|
| Phase 2 WB/BB test sets | Forbidden | Direct test leakage, so the X/Y/Z scores become optimistic. |
| Attacked train/val made with the Phase 2 attacks | Forbidden | The setting is fitted to the exact attack and budget. Even a single number (q, σ, k) picked by attacked accuracy is a fit to the attack. For BB it also breaks the threat model. |
| Other attacks on val (FGSM, CW, other patch attacks) | Forbidden for selection | Same threat family, so it still leaks the attack type. It could be a clearly labelled post-hoc sensitivity analysis, but it is never used to choose a setting. |
| Clean val with a clean-accuracy budget | **Used** | Attack-agnostic. Each preprocessing defence has one setting with a monotone strength order (lower JPEG quality, larger σ, larger kernel all remove more detail). The rule is "strongest setting that keeps clean accuracy within budget", which needs no attack. |
| Literature defaults with no tuning | Acceptable fallback | Zero leakage. The grids below contain these defaults (JPEG q = 75, median k = 3–5, small σ). |
| Clean test | Forbidden for tuning | Evaluation only. |

**Conclusion.** Tuning is feasible for every planned method, using clean val only. Clean val shows what a setting costs on clean images, not how much it buys against attacks. The robustness side therefore comes from the strength order of each defence, not from attacked data. This is a deliberate trade-off: the chosen settings may be weaker than the "best against PGD/Pixle" settings, but the test scores are honest.

## 4. Planned methods

### 4.1 Preprocessing defences (test time, same frozen Phase 1 model)

These follow the Phase 3 diagram: 3 test sets × {no defence, D1, D2, D3} = 12 pairings. Code: `modules/preprocessing/defences.py`.

| | Defence | Method | Setting and grid (mildest first) | Why it is attack-agnostic |
|---|---|---|---|---|
| D1 | JPEG compression | Encode and decode the input as JPEG at quality q | q ∈ {90, 75, 60, 50, 40, 30} | Generic removal of high-frequency detail (Das et al. 2017; Guo et al. 2018). The grid is the usual JPEG quality range. |
| D2 | Gaussian randomized smoothing | n = 32 copies with N(0, σ²) noise; the prediction is the majority vote (Cohen et al. 2019). Fixed seed 4266, no abstain | σ ∈ {4, 8, 12, 16, 24} (0–255 scale) | Averages the model over random noise, so the decision depends less on any small input change. σ is limited by the clean-accuracy budget, not by an attack budget. |
| D3 | Median smoothing | k × k median filter per channel | k ∈ {3, 5, 7} | Standard edge-preserving denoiser for impulse and local corruption. It is a different family from D1 (frequency domain) and D2 (noise averaging). |

Median smoothing is a preprocessing defence, not an augmentation defence: it changes nothing in training. It was moved out of `modules/augmentation-based/` for this reason.

Methods considered and not chosen for D3:
- **Bit-depth reduction:** little extra over D1 and D3.
- **Total-variation minimisation:** slow, and overlaps D2.
- **Random resize and pad:** mainly aimed at gradient-based attacks.

Notes:
- **D1:** the clean crops and the Pixle images are already JPEG q95, so q = 90 is close to an identity on them. That is why the grid goes down to 30.
- **D2:** the Phase 1 model was never trained with noise, so clean accuracy will fall quickly as σ grows and the budget will likely cap σ low. No certified radius is claimed; D2 is an empirical defence here. n and the seed are fixed before any run and are not tuned.

### 4.2 Augmentation-based defences (training time, retrained models)

| Defence | Method | Grid | Frozen choice (clean val) |
|---|---|---|---|
| Random erasing | With p = 0.5, one random rectangle (area 0.02 to max_area, aspect 0.3–3.3) is replaced with uniform noise during training | max_area ∈ {0.1, 0.2, 0.3} | max_area = 0.1 (val acc 0.9961, macro F1 0.998) |
| Mixup | Blend image pairs and their labels, λ ~ Beta(α, α) | α ∈ {0.1, 0.2, 0.4} | α = 0.1 (val acc 1.0, macro F1 1.0) |

Both train on `train/` only, using the Phase 1 recipe. Both pick their checkpoint and setting on clean `val/` macro F1. They are already trained, and their choices are in `tuned_defences.yaml`. These defences produce new models, so they do not fit the diagram's "same Phase 1 model" comparison and get their own table (section 5).

### 4.3 Excluded: TRADES and all adversarial training

TRADES was removed from Phase 3, for two reasons:
- **It generates adversarial data:** PGD examples are made from `train/` on the fly during training.
- **Its budget came from the attack:** ε = 8/255 and step 2/255 were copied from the Phase 2 white-box PGD.

That is both data leakage and knowledge leakage under section 2, and its WB score would be close to in-distribution. PGD-AT and Fast-AT are excluded for the same reason. The script, its `run_all.sh` step, its `tuned_defences.yaml` entry and the `adv_*` settings in `config.yaml` have been removed.

## 5. Tuning and evaluation protocol

**Tuning (fixed before running):**
1. **Preprocessing** (`tune_preprocessing.py`): measure clean-val accuracy with no defence, then for each grid setting. Choose the **strongest setting whose accuracy drop is at most 1 point**. If none qualifies, use the mildest setting and set `within_tolerance: false`. The choices are frozen in `tuned_defences.yaml`. Outputs go to the directories in section 5.1.
2. **Augmentation** (training scripts): best clean-val macro F1, ties going to the milder setting. This is unchanged.

**Evaluation**: one script per defence group. Each runs after tuning, with frozen settings only. `--defence` selects which defences to test (default `all`), and no defence is always included as the baseline.
- `eval_preprocessing.py --defence {jpeg, gaussian, median, all}`: the preprocessing defences on the Phase 1 model. With `all` this gives the 12 pairings.
- `eval_augmentation.py --defence {random_erasing (or erasing), mixup, all}`: the retrained models against the Phase 1 model. With `all` this gives 9 pairings.
- A run with only some defences replaces their rows in the results and keeps the rest, so the summary always covers every defence tested so far.
- For each pairing it records accuracy with a 95% Wilson CI, macro F1, n, and an exact McNemar p-value against no defence on the same images. For the clean test set it also records accuracy on the 130 images that the attacked sets were made from.
- **Selection:** worst case W_i = min(X_i, Y_i, Z_i) (clean, WB, BB accuracy). The defence with the largest W_i is chosen. No defence / Phase 1 is the reference baseline, not a candidate.
- Output: `test_results/<group>/results.csv` and `summary.csv`.

### 5.1 Output directories

Each defence group gets its own directory under `checkpoints/` (tuning runs) and under `tuning_logs/` (tuning record). The scripts create these directories if they are missing, and never write one group's files into another group's directory. The preprocessing defences follow the same layout as the augmentation defences.

| Group | Tuning runs (`checkpoints/`) | Tuning record (`tuning_logs/`) | Final evaluation |
|---|---|---|---|
| Preprocessing (D1–D3) | `checkpoints/preprocessing/`: `<defence>_grid.csv` for each tuning run, `logs/tune_preprocessing.log` | `tuning_logs/preprocessing/`: `<defence>_grid.csv` (clean-val accuracy, macro F1, loss and drop for each setting, with no defence as the first row), `summary.csv` (one row per defence: chosen setting, val accuracy, drop, within tolerance) | `test_results/preprocessing/`: `results.csv`, `summary.csv` |
| Augmentation (random erasing, Mixup) | `checkpoints/augmentation/`: `<defence>_<param><value>/best.keras` + `history.csv`, `<defence>_grid.csv`, `logs/` | `tuning_logs/augmentation/`: per-epoch histories, `grid.csv` for each defence, `summary.csv`, `loss_report.md`, loss plots | `test_results/augmentation/`: `results.csv`, `summary.csv` |

`tuned_defences.yaml` holds the frozen setting of every defence and is the only file the evaluation scripts read settings from.

## 6. Risks and limitations

- **Val resolution.** Val has 257 images, so 1 point is about 2.6 images, and settings whose scores differ by less than the tolerance are statistically tied. The fixed rule ("strongest within budget") breaks the tie the same way every time instead of chasing noise.
- **Optimistic clean val.** Clean val was already used to pick the Phase 1 checkpoint and to tune the Phase 2 PGD settings. The no-defence val accuracy is close to 1.0, which makes the 1-point budget relative to a slightly optimistic baseline. No attacked pixels are involved, so this is not attack leakage.
- **The final pick is made on test.** The brief chooses the winner by argmax of the worst case over the test sets. With only three candidates the selection bias is small, but the winner's score is a *selected* score, not an unbiased estimate.
- **Non-adaptive attacks.** The WB images were crafted against the undefended model, so the preprocessing defences face a gray-box attack that does not know about them. The results do not show robustness to adaptive attacks such as BPDA or EOT (Athalye et al. 2018). That is out of scope.
- **Subset.** WB/BB contain only the 130 images Phase 1 got right, so WB/BB accuracy is accuracy among images that were originally correct. Clean accuracy is reported on both the full 131 and the 130 subset.
- **Randomness in D2.** Predictions depend on the noise draw. A fixed seed makes them reproducible, but a different seed can move accuracy by about one image.

## 7. Acceptance criteria

- [x] No training or tuning script reads `pgd_attacked/`, `pixle_attacked/` or any attacked folder, and `check_tuning_split()` rejects them.
- [x] No defence setting is taken from a Phase 2 attack setting. TRADES and the `adv_*` keys are removed.
- [x] D1–D3 are implemented as test-time functions behind one predictor interface (`make_predictor`).
- [ ] `tune_preprocessing.py` writes 3 grid CSVs to both `checkpoints/preprocessing/` and `tuning_logs/preprocessing/`, `tuning_logs/preprocessing/summary.csv`, and 3 frozen entries in `tuned_defences.yaml`. Needs the Phase 1 checkpoint.
- [ ] `eval_preprocessing.py --defence all` writes 12 rows, and the no-defence row reproduces Phase 1/2: clean 0.9924, BB about 0.377 (Pixle success rate 0.623).
- [ ] `eval_augmentation.py --defence all` writes 9 rows.
- [ ] Both `summary.csv` files name the chosen defence and its worst case.

## 8. How to run

```bash
cd src/Phase_3
python tune_preprocessing.py --model ../Phase_1/runs/tf_resnet50_speed_limit_aug/best/best.keras
python eval_preprocessing.py --defence all        # or e.g. --defence jpeg median
python eval_augmentation.py --defence all         # or e.g. --defence mixup
```

Or run everything that trains or tunes with `bash modules/augmentation-based/run_all.sh`. Evaluation is run separately, on purpose.
