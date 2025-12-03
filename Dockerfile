# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

# System deps for Pillow, CairoSVG (cairo + pango + fonts)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libcairo2 \
    libcairo2-dev \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libfontconfig1 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy source
COPY . /workspace

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 12.1 wheels (compatible with Runpod 12.x)
# This index serves cu121 wheels; CPU fallback isn't needed on GPU runtimes.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# Core Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Default ENV (can be overridden by Runpod env settings)
ENV WEIGHT_PATH=/runpod-volume/OmniSVG \
    WEIGHT_PATH_4B=/runpod-volume/OmniSVG1.1_4B \
    WEIGHT_PATH_8B=/runpod-volume/OmniSVG1.1_8B \
    CONFIG_PATH=/workspace/config.yaml \
    QWEN_LOCAL_DIR=/runpod-volume/Qwen2.5-VL-3B-Instruct \
    QWEN_MODEL_4B=/runpod-volume/Qwen2.5-VL-3B-Instruct \
    QWEN_MODEL_8B=/runpod-volume/Qwen2.5-VL-7B-Instruct \
    SVG_TOKENIZER_CONFIG=/workspace/config.yaml \
    ENABLE_DUMMY=true

# Serverless entrypoint
CMD ["python", "-u", "handler.py"]
