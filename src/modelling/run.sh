cd /app

# python -u train_convae.py
# python -u train.py
python test.py
# python -u gradcam.py --model /app/runs/best_robust.keras --predictions /app/results/test_robust/predictions.csv --output-dir /app/results/gradcam_test_robust
# python -u gradcam.py --model /app/runs/best_robust.keras --predictions /app/results/pixle_attacked_robust/predictions.csv --output-dir /app/results/gradcam_pixle_attacked_robust
# python -u gradcam.py --model /app/runs/best_robust.keras --predictions /app/results/pgd_attacked_robust/predictions.csv --output-dir /app/results/gradcam_pgd_attacked_robust

# echo "working!"

# pip list