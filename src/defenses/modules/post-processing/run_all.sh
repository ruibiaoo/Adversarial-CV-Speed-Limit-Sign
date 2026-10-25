#!/usr/bin/env bash
# Tune the post-processing defences (gaussian_noise, high_confidence, reverse_sigmoid) on the
# val split named in data.yaml, then test them alone and in every ordered chain on the clean
# and attacked test sets. Tuning reads only val; testing reads only the test sets.
# Run from src/defenses/:  docker compose run --rm phase3 bash modules/post-processing/run_all.sh
# Results: modules/post-processing/tuned_defences.yaml, tuning_logs/post-processing/,
# testing_results/post_processing_based/. Logs: testing_results/post_processing_based/logs/<step>.log.
set -euo pipefail
LOGS=testing_results/post_processing_based/logs
mkdir -p "$LOGS"
D=modules/post-processing

python $D/tune_postprocessing.py 2>&1 | tee "$LOGS/tune.log"
python $D/test_postprocessing.py 2>&1 | tee "$LOGS/test.log"
echo "ALL DONE"
