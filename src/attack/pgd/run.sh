#!/bin/bash
set -e
cd /repo/src/attack/pgd

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

OVERWRITE_ARG=""
if [ "${OVERWRITE,,}" = "true" ] || [ "${OVERWRITE}" = "1" ]; then
    OVERWRITE_ARG="--overwrite"
fi

echo "=========================================================="
echo "Starting PGD Adversarial Example Generation"
echo "Splits: ${SPLITS:-train val}"
echo "Skip Misclassified: ${SKIP_MISCLASSIFIED:-false}"
echo "Max Images: ${MAX_IMAGES:-all}"
echo "=========================================================="

python -u pgd_attacked_images_generation.py \
    ${SPLIT_ARGS} \
    ${SKIP_FLAG} \
    ${MAX_IMAGES_ARG} \
    ${OVERWRITE_ARG}

