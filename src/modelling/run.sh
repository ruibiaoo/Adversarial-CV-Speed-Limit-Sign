set -e
cd /app

# python -u train_convae.py
python -u train.py
for split in val test pgd_attacked pixle_attacked; do
    python -u test.py \
        --split "$split" \
        --output-dir "/app/results/grl_adversarial_retraining/$split"
done
# python -u gradcam.py --model /app/runs/best_robust.keras --predictions /app/results/test_robust/predictions.csv --output-dir /app/results/gradcam_test_robust
# python -u gradcam.py --model /app/runs/best_robust.keras --predictions /app/results/pixle_attacked_robust/predictions.csv --output-dir /app/results/gradcam_pixle_attacked_robust
# python -u gradcam.py --model /app/runs/best_robust.keras --predictions /app/results/pgd_attacked_robust/predictions.csv --output-dir /app/results/gradcam_pgd_attacked_robust

# echo "working!"

# pip list
