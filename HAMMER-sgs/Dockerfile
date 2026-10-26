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

# Install torch from the PyTorch index first: it is much faster than PyPI from our network,
# and a CUDA 12.x build runs on older cluster drivers than PyPI's default CUDA 13 wheels.
# requirements.txt then sees torch/torchvision as already satisfied.
RUN pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cu126 torch torchvision

# Install the remaining dependencies so code changes don't invalidate this layer.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY scripts/ scripts/

CMD ["bash"]
