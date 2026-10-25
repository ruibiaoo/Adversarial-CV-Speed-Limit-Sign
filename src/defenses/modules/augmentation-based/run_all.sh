#!/usr/bin/env bash
# Train the augmentation-based defences (random erasing, Mixup), then write the
# loss report to tuning_logs/augmentation/. The preprocessing defences are not
# tuned: test_defences.py tests them with fixed settings.
# No step reads a Phase 2 attacked image (PRD_defences.md).
# Run from Phase_3/:  docker compose run --rm phase3 bash modules/augmentation-based/run_all.sh
# Logs: checkpoints/augmentation/logs/<step>.log. Results: checkpoints/augmentation/,
# tuning_logs/augmentation/, and the frozen settings in modules/augmentation-based/tuned_defences.yaml.
set -euo pipefail  # finished runs are reused, so re-running resumes after a crash
LOGS=checkpoints/augmentation/logs
mkdir -p "$LOGS" tuning_logs/augmentation
D=modules/augmentation-based

python $D/random_erasing.py 2>&1 | tee "$LOGS/random_erasing.log"
python $D/mixup.py          2>&1 | tee "$LOGS/mixup.log"

python $D/report.py         2>&1 | tee "$LOGS/report.log"
echo "ALL DONE"
