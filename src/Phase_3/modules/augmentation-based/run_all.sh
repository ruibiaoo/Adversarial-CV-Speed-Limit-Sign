#!/usr/bin/env bash
# Train / calibrate the four chosen augmentation-based defences, one after another,
# then write the loss report to tuning/augmentation-tuning/.
# Run from Phase_3/:  docker compose run --rm phase3 bash modules/augmentation-based/run_all.sh
# Logs: runs/logs/<step>.log. Results: runs/*_grid.csv, defence_configs.yaml, tuning/augmentation-tuning/.
set -euo pipefail  # finished runs are reused, so re-running resumes after a crash
mkdir -p runs/logs
D=modules/augmentation-based

python $D/random_erasing.py 2>&1 | tee runs/logs/random_erasing.log
python $D/mixup.py          2>&1 | tee runs/logs/mixup.log
python $D/trades.py         2>&1 | tee runs/logs/trades.log

# Median smoothing trains nothing: it is calibrated on the frozen Phase 1 checkpoint.
# Run last, and skipped (with a warning) if the checkpoint is not on this machine.
P1=$(python -c "from modules.common import load_config; print(load_config()['phase1_model'])")
if [ -f "$P1" ]; then
  [ -f runs/median_smoothing_grid.csv ] || python $D/median_smoothing.py --model "$P1" 2>&1 | tee runs/logs/median_smoothing.log
else
  echo "WARNING: Phase 1 checkpoint not found ($P1), skipping median smoothing."
  echo "Copy it there, then run: python $D/median_smoothing.py"
fi

python $D/report.py         2>&1 | tee runs/logs/report.log
echo "ALL DONE"
