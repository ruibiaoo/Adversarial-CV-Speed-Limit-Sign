# Phase 2: Pixle black-box attack

This phase attacks the Phase 1 speed-limit-sign classifier using Pixle, an
`L0` black-box attack based on rearranging pixels. The implementation only
requires a probability oracle: it does not use gradients, model weights, or
the training procedure.

## Files

- `pixle_attack.py` contains the reusable Pixle implementation.
- `evaluate.py` adapts a TensorFlow/Keras model and evaluates the attack on
  `data/data_updated/test`.
- `test_pixle_attack.py` contains lightweight tests for query accounting and
  initial-query reuse.

## Default attack

The defaults match the paper's main setup:

```text
algorithm: Restart-Iterative
mapping: random
patch side: 3
operation: overwrite
restarts: 100
iterations per restart: 50
```

Each attack begins with one model query. Each candidate image costs one more
query. The attack stops early when an initially correct image is misclassified.
Images the model initially misclassifies are skipped and do not count toward
the attack success-rate denominator.

## Run tests

From the repository root:

```bash
PYTHONPATH=src python -m unittest attack.pixle.test_pixle_attack
```

## Evaluate the Phase 1 Keras model

The model path below is the default path used inside the Phase 1 Docker
container. Replace it with the local path to `best.keras` if running outside
Docker.

```bash
python src/attack/pixle/evaluate.py \
  --model PATH_TO_BEST_KERAS \
  --data data/data_updated \
  --image-size 256 \
  --results-dir src/attack/pixle/results \
  --adversarial-dir data/data_updated/pixle_attacked
```

Outputs:

```text
src/attack/pixle/results/pixle_results.csv
src/attack/pixle/results/pixle_metrics.json
data/data_updated/pixle_attacked/<original_class>/*.png
```

The CSV reports the actual class, initial prediction, adversarial prediction,
success, model queries, and final `L0` distance for every image. Every
initially correct image that is attacked is saved, including unsuccessful
attacks. Successful attacks save the successful adversarial image; unsuccessful
attacks save the last perturbed candidate. Images are grouped by their original
class, and the CSV records the saved image path and type. Saved adversarial
examples are written as PNG files. The metrics JSON
reports attack success rate over initially correct images, mean queries, and
mean `L0`.

During evaluation, a progress bar shows the number of model queries used for
the current image. It starts at one for the initial query and advances for
each Pixle candidate. Use `--no-progress` to disable it.

## Change attack parameters

For example, to compare patch sizes and mapping functions:

```bash
python src/attack/pixle/evaluate.py --model PATH_TO_BEST_KERAS \
  --patch-size 1 --mapping random --restarts 100 --iterations 50

python src/attack/pixle/evaluate.py --model PATH_TO_BEST_KERAS \
  --patch-size 3 --mapping distance_distribution \
  --restarts 100 --iterations 50
```

Supported mappings are `random`, `similarity`, `distance`,
`similarity_distribution`, and `distance_distribution`. Patch sizes may be
specified as `3` or `3x5`. The alternative `iterative` algorithm and `swap`
operation are also available for ablation studies.

## Experimental protocol

Use a validation subset to choose attack parameters. Keep the held-out test
set for the final comparison. Only images correctly classified before the
attack should be included in the attack-success denominator; otherwise the
baseline's existing errors would inflate the attack result.
