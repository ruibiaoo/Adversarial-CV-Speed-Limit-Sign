cd /app

# python -u train.py
python test.py
# python -u gradcam.py --model /app/runs/best.keras --predictions /app/results/test/predictions.csv --output-dir /app/results/gradcam --max-images 50
# python -u gradcam.py --max-images 20 --model /app/runs/best.keras --predictions /app/results/test/predictions.csv --output-dir /app/results/gradcam
# echo "working!"

# pip list