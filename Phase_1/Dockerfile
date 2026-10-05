# Base image: TF runtime with CUDA support
# Works with NVIDIA RTX 30/40-series and other modern CUDA GPUs
FROM tensorflow/tensorflow:2.15.0-gpu

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    OMP_NUM_THREADS=4

# Set working directory inside container
WORKDIR /app

# Upgrade pip and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source scripts into container
COPY train.py run.sh config.yaml tuner.yaml ./
COPY modules/ ./modules/
RUN chmod +x run.sh

# Default execution entrypoint
CMD ["bash", "run.sh"]
