FROM tensorflow/tensorflow:2.16.2-gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=4 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /repo/src/attack/pgd

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./
RUN sed -i 's/\r$//' run.sh 2>/dev/null && chmod +x run.sh 2>/dev/null || true

CMD ["bash", "run.sh"]

