#!/usr/bin/env bash
# Train / calibrate the four chosen augmentation-based defences, one after another,
# then write the loss report to tuning/augmentation-tuning/.
# Run from Phase_3/:  docker compose run --rm phase3 bash modules/augmentation-based/run_all.sh
# Logs: runs/logs/<step>.log. Results: runs/*_grid.csv, defence_configs.yaml, tuning/augmentation-tuning/.
set -euo pipefail  # finished runs are reused, so re-running resumes after a crash
mkdir -p runs/logs
D=modules/augmentation-based

# No-defence baseline: the reference curve in the loss report.
python $D/baseline.py 2>&1 | tee runs/logs/none.log

# Median smoothing is calibrated on a frozen model: the Phase 1 checkpoint if
# present, otherwise the baseline above (same recipe).
P1=$(python -c "from modules.common import load_config; print(load_config()['phase1_model'])")
MODEL=runs/none/best.keras
if [ -f "$P1" ]; then MODEL="$P1"; else echo "Phase 1 checkpoint not found ($P1), using $MODEL"; fi
[ -f runs/median_smoothing_grid.csv ] || python $D/median_smoothing.py --model "$MODEL" 2>&1 | tee runs/logs/median_smoothing.log

python $D/random_erasing.py 2>&1 | tee runs/logs/random_erasing.log
python $D/mixup.py          2>&1 | tee runs/logs/mixup.log
python $D/trades.py         2>&1 | tee runs/logs/trades.log
python $D/report.py         2>&1 | tee runs/logs/report.log
echo "ALL DONE"
