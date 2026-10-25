# Post-processing defences

Test-time defences that transform the base classifier's **output probability vector**
instead of the image. The base model (`best_robust.keras`) is never retrained. Built on the
[Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)
(ART) postprocessors.

## The three defences

| Defence | ART class | Grid (mild → strong) | What it does |
|---|---|---|---|
| `gaussian_noise` | `GaussianNoise` | `scale ∈ {0.01, 0.05, 0.1, 0.2, 0.3}` | Adds `N(0, scale²)` noise to the probability vector, then renormalises. |
| `high_confidence` | `HighConfidence` | `cutoff ∈ {0.1, 0.25, 0.5, 0.75, 0.9}` | Zeros every probability below `cutoff`. |
| `reverse_sigmoid` | `ReverseSigmoid` | `gamma ∈ {0.1, 0.2, 0.3, 0.5}`, `beta = 1.0` | Adds a reverse-sigmoid perturbation that scrambles the vector while keeping the top class (Lee et al., 2019). |

The base model's softmax is computed once; the postprocessors then compose on the `(N, C)`
array, so the tester runs each defence alone and **every ordered chain of two and three**
(`noise`, `highconf`, `revsig`, `noise_then_highconf`, …). Order matters — e.g. thresholding
before vs after adding noise gives different vectors — which is why permutations are tested.

## Why `Rounded` is excluded (3 of ART's 4)

ART's fourth postprocessor, `Rounded`, only truncates each probability to a number of
decimals. That never changes which class is largest, so it cannot change accuracy on any
test set and gives a score-based attacker almost as much information as the raw vector. It is
the weakest of the four and is left out; the three above all genuinely distort the output.

## Tuning objective (why it is not "best macro F1")

Because these defences change the output vector and not the image, the top-1 class — and so
accuracy and macro F1 — barely moves, whether the input is clean or already attacked.
Selecting on "best val macro F1" would therefore pick the mildest setting every time and
measure nothing useful.

The real job of a post-processing defence is to **deny a score-based / query-based attacker
(e.g. Pixle) the fine-grained output it needs**, while not hurting the honest user's accuracy.
So `tune_postprocessing.py` selects, per defence, the setting that **maximises output
obfuscation** — mean top-1 confidence drop plus mean output-entropy increase vs the undefended
base — **among the settings whose val macro F1 stays within a tolerance `τ` (default 0.01) of
the undefended baseline**. This is justifiable and leakage-free: it is tuned only on the val
split named in `data.yaml` (clean or attacked val), never on the test split, and it encodes the
actual security goal (obfuscate the output at no accuracy cost).

## data.yaml drives the splits

The train/val/test folders and the class order come from `<data root>/data.yaml` (the YOLO
spec, read by `modules/config_loader.py`). To tune against an attack, point `val:` in
`data/data_updated/data.yaml` at an attacked-val folder and re-run the tuner; the test splits
(clean `test`, `pgd_attacked`, `pixle_attacked`) are configured in `../../config.yaml` and are
only read by the tester.

## Running (Docker)

From `src/defenses/`:

```bash
# tune on val, then test alone + all chains on clean + attacked test sets
docker compose run --rm phase3 bash modules/post-processing/run_all.sh

# or individually
docker compose run --rm phase3 python modules/post-processing/tune_postprocessing.py
docker compose run --rm phase3 python modules/post-processing/test_postprocessing.py
docker compose run --rm phase3 python modules/post-processing/tune_postprocessing.py --defence noise --tolerance 0.02
```

Outputs:
- `modules/post-processing/tuned_defences.yaml` — frozen chosen setting per defence.
- `tuning_logs/post-processing/<defence>/grid.csv` — per-setting val scores + obfuscation.
- `testing_results/post_processing_based/<chain>/<test set>/` — the five Phase-2 report files.
- `testing_results/post_processing_based/summary.{csv,md}` — ranked table + clean-split
  obfuscation table.
