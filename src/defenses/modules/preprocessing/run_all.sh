#!/usr/bin/env bash
# Tune the preprocessing defences (jpeg, median, gaussian) on the val split named in data.yaml,
# then test no_defence and every ordered chain of the tuned defences on the clean and attacked
# test sets. Tuning reads only val; testing reads only the test sets.
# Run from src/defenses/:  docker compose run --rm phase3 bash modules/preprocessing/run_all.sh
# Results: modules/preprocessing/tuned_defences.yaml, tuning_logs/preprocessing/,
# testing_results/preprocessing_based/. Logs: testing_results/preprocessing_based/logs/<step>.log.
set -euo pipefail
LOGS=testing_results/preprocessing_based/logs
mkdir -p "$LOGS"
D=modules/preprocessing

python $D/tune_preprocessing.py 2>&1 | tee "$LOGS/tune.log"
python $D/test_defences.py 2>&1 | tee "$LOGS/test.log"
echo "ALL DONE"
