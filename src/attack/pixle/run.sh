#!/bin/bash
set -e
cd /repo

echo "Running Pixle unit tests..."
python -m unittest src/attack/pixle/test_pixle_attack.py

SKIP_FLAG="--no-skip-misclassified"
if [ "${SKIP_MISCLASSIFIED,,}" = "true" ] || [ "${SKIP_MISCLASSIFIED}" = "1" ]; then
    SKIP_FLAG="--skip-misclassified"
fi

SPLIT_ARGS=""
if [ -n "${SPLITS}" ]; then
    SPLIT_ARGS="--splits ${SPLITS}"
fi

MAX_IMAGES_ARG=""
if [ -n "${MAX_IMAGES}" ]; then
    MAX_IMAGES_ARG="--max-images ${MAX_IMAGES}"
fi

WORKERS_ARG="--workers ${WORKERS:-8}"

echo "=========================================================="
echo "Starting Pixle Adversarial Example Generation (GPU)"
echo "Splits: ${SPLITS:-train val test}"
echo "Skip Misclassified: ${SKIP_MISCLASSIFIED:-false}"
echo "Workers: ${WORKERS:-8}"
echo "Max Images: ${MAX_IMAGES:-all}"
echo "=========================================================="

python -u src/attack/pixle/evaluate.py \
    --model src/modelling/runs/best_baseline.keras \
    --data data/data_updated \
    --adversarial-dir data/data_updated_pixle \
    --results-dir src/attack/pixle/results \
    ${SPLIT_ARGS} \
    ${SKIP_FLAG} \
    ${WORKERS_ARG} \
    ${MAX_IMAGES_ARG} \
    --no-progress
