# 银发守护者系统 Docker 镜像
# 适用场景: Linux 部署 (需 NVIDIA GPU + nvidia-container-toolkit)
# 构建: docker build -t silver-guardian .
# 运行: docker run --gpus all -v ./data:/app/data -v ./models:/app/models silver-guardian

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.11 python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY config/ config/
COPY data/face_db/ data/face_db/
COPY scripts/ scripts/

RUN mkdir -p data/logs data/alarms

CMD ["python3", "src/main.py"]
