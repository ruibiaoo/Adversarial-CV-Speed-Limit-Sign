#!/usr/bin/env bash
# Train the augmentation-based defences (random erasing, Mixup), calibrate the
# preprocessing defences (JPEG, Gaussian smoothing, median) on clean val, then
# write the loss report to tuning_logs/augmentation/.
# No step reads a Phase 2 attacked image (PRD_defences.md).
# Run from Phase_3/:  docker compose run --rm phase3 bash modules/augmentation-based/run_all.sh
# Logs: checkpoints/<group>/logs/<step>.log. Results: checkpoints/augmentation/, checkpoints/preprocessing/,
# tuning_logs/augmentation/, tuning_logs/preprocessing/, tuned_defences.yaml.
set -euo pipefail  # finished runs are reused, so re-running resumes after a crash
LOGS=checkpoints/augmentation/logs
PRE_LOGS=checkpoints/preprocessing/logs
mkdir -p "$LOGS" "$PRE_LOGS" tuning_logs/augmentation tuning_logs/preprocessing
D=modules/augmentation-based

python $D/random_erasing.py 2>&1 | tee "$LOGS/random_erasing.log"
python $D/mixup.py          2>&1 | tee "$LOGS/mixup.log"

# Preprocessing defences train nothing: they are calibrated on the frozen Phase 1 checkpoint.
# Skipped (with a warning) if the checkpoint is not on this machine.
P1=$(python -c "from modules.config_loader import load_config; print(load_config()['phase1_model'])")
if [ -f "$P1" ]; then
  python tune_preprocessing.py --model "$P1" 2>&1 | tee "$PRE_LOGS/tune_preprocessing.log"
else
  echo "WARNING: Phase 1 checkpoint not found ($P1), skipping preprocessing calibration."
  echo "Copy it there, then run: python tune_preprocessing.py"
fi

python $D/report.py         2>&1 | tee "$LOGS/report.log"
echo "ALL DONE"
