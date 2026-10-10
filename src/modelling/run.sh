cd /app

# python -u train.py
# python test.py
# python -u gradcam.py --model /app/runs/best.keras --predictions /app/results/test/predictions.csv --output-dir /app/results/gradcam --max-images 50
python -u gradcam.py --max-images 130 --model /app/runs/best.keras --predictions /app/results/phase2_pgd_attacked/predictions.csv --output-dir /app/results/gradcam_pgd_attacked
# echo "working!"

# pip list