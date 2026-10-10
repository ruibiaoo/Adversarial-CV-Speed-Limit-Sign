cd /app

# python -u train.py
# python test.py
# python -u gradcam.py --model /app/runs/best.keras --predictions /app/results/test/predictions.csv --output-dir /app/results/gradcam --max-images 50
python -u gradcam.py --model /app/runs/best.keras --predictions /app/results/test_new/predictions.csv --output-dir /app/results/gradcam_test_new
python -u gradcam.py --model /app/runs/best.keras --predictions /app/results/val_new/predictions.csv --output-dir /app/results/gradcam_val_new

# echo "working!"

# pip list