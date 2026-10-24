# Base image: TF runtime (CPU)
# FROM tensorflow/tensorflow:2.16.2
# GPU base image (for CUDA/GPU systems):
FROM tensorflow/tensorflow:2.16.2-gpu

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4
# For GPU:
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set working directory inside container
WORKDIR /app

# Install Python packages into the base image's system Python (TF already present)
COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /uvx /bin/
COPY src/modelling/requirements.txt ./requirements.txt
RUN uv pip install --system -r requirements.txt

# Copy source scripts into container
COPY src/modelling/train.py src/modelling/train_convae.py src/modelling/test.py src/modelling/gradcam.py src/modelling/run.sh ./
COPY src/modelling/modules/ ./modules/
COPY src/modelling/configs/ ./configs/
RUN sed -i 's/\r$//' run.sh && chmod +x run.sh


# Default execution entrypoint
CMD ["bash", "run.sh"]
