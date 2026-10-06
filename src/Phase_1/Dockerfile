# Base image: TF runtime with CUDA support
# Works with NVIDIA RTX 30/40-series and other modern CUDA GPUs
FROM tensorflow/tensorflow:2.16.2-gpu

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    OMP_NUM_THREADS=4

# Set working directory inside container
WORKDIR /app

# Install Python packages
COPY --from=ghcr.io/astral-sh/uv:0.11.31 /uv /uvx /bin/
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

ENV PATH="/app/.venv/bin:$PATH"

# Copy source scripts into container
COPY src/Phase_1/train.py src/Phase_1/test.py src/Phase_1/run.sh ./
COPY src/Phase_1/modules/ ./modules/
COPY src/Phase_1/configs/ ./configs/
RUN chmod +x run.sh

# Default execution entrypoint
CMD ["bash", "run.sh"]
