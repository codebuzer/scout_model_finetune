FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
ENV FORCE_CUDA="1"
ENV BNB_CUDA_VERSION=122
ENV CUDA_VISIBLE_DEVICES=0

# Symlinks for CUDA libraries
RUN ln -sf /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so && \
    ln -sf /usr/local/cuda/lib64/libnvrtc.so /usr/lib/libnvrtc.so && \
    ln -sf /usr/local/cuda/lib64/libcusparse.so.12 /usr/local/cuda/lib64/libcusparse.so && \
    ldconfig

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    ninja-build \
    wget \
    cmake \
    pkg-config \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirement files
COPY requirements-base.txt requirements-cuda.txt ./

# Core Python setup
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir numpy==1.24.3 scipy==1.11.3 ninja

# Install PyTorch with CUDA 12.1 (compatible with CUDA 12.2)
RUN pip3 install --no-cache-dir torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install bitsandbytes
RUN pip3 uninstall -y bitsandbytes && \
    pip3 install bitsandbytes

# Install FlashAttention v2.3.3 (compatible with torch 2.1.0)
RUN pip3 install --no-cache-dir packaging wheel && \
    git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout v2.3.3 && \
    MAX_JOBS=4 TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9" pip3 install --no-build-isolation . && \
    cd .. && \
    rm -rf flash-attention

# Install additional base requirements
RUN pip3 install --no-cache-dir -r requirements-base.txt

# Install CUDA-specific requirements
RUN pip3 install --no-cache-dir -r requirements-cuda.txt

# Create app folders
RUN mkdir -p /app/models /app/data /app/logs ~/.huggingface

# Copy source code and models
COPY src/ /app/src/
COPY models/ /app/models/

# Entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Final environment sanity check (CPU-safe)
RUN python3 -c "import numpy; print('NumPy version:', numpy.__version__)" && \
    python3 -c "import torch; print('PyTorch version:', torch.__version__)" && \
    python3 -c "import transformers; print('Transformers version:', transformers.__version__)"

# No need to run bitsandbytes in the Dockerfile, you can test it later in entrypoint or after container starts
