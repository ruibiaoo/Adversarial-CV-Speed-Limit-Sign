# Base image: PyTorch runtime with CUDA 12.4 and cuDNN 9 support
# Compatible with NVIDIA RTX 30/40-series and other modern CUDA-enabled GPUs
# FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

FROM ultralytics/ultralytics:8.4.146



# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    OMP_NUM_THREADS=4

# Install system dependencies required for OpenCV, Git, and Ultralytics
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     git \
#     curl \
#     ca-certificates \
#     libgl1 \
#     libglib2.0-0 \
#     libgomp1 \
#     && rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Upgrade pip and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir ultralytics==8.4.146 "ray[tune]" optuna wandb
# Copy source scripts into container
COPY train.py run.sh config.yaml tuner.yaml ./
COPY modules/ ./modules/
RUN chmod +x run.sh

# Default execution entrypoint
CMD ["bash", "run.sh"]
