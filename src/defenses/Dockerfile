# Same TF as src/modelling/Dockerfile, so the base model (saved with Keras 3) loads as is.
FROM tensorflow/tensorflow:2.16.2-gpu
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
WORKDIR /repo/src/defenses
# modelling/requirements.txt + scikit-image (build context is src/)
COPY modelling/requirements.txt /tmp/req1.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/req1.txt "scikit-image>=0.21.0"
