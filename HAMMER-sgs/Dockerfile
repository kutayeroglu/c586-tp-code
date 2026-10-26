FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System libs needed by opencv-python / ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install dependencies first so code changes don't invalidate this layer.
# The default PyPI torch wheels on Linux ship with CUDA; run with --gpus all to use the GPU.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY scripts/ scripts/

CMD ["bash"]
