FROM tensorflow/tensorflow:2.15.0-gpu
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
WORKDIR /repo/Phase_3
# Phase_1/requirements.txt + scikit-image (build context is the repo root)
COPY Phase_1/requirements.txt /tmp/req1.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/req1.txt "scikit-image>=0.21.0"
