FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"
ENV FORCE_CUDA="1"

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

# Set working directory
WORKDIR /app

# Copy requirements files
COPY requirements-base.txt requirements-cuda.txt ./

# Install Python packages in specific order with exact versions
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir numpy==1.24.3 && \
    pip3 install --no-cache-dir scipy==1.11.3 && \
    pip3 install --no-cache-dir ninja && \
    pip3 install --no-cache-dir torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir bitsandbytes==0.41.1 && \
    pip3 install --no-cache-dir transformers==4.51.0 && \
    pip3 install --no-cache-dir accelerate==0.27.0 && \
    pip3 install --no-cache-dir einops==0.6.1 && \
    pip3 install --no-cache-dir -r requirements-base.txt

# Try installing flash-attention from source
RUN pip3 install --no-cache-dir packaging wheel && \
    git clone https://github.com/Dao-AILab/flash-attention.git && \
    cd flash-attention && \
    git checkout v2.3.3 && \
    pip3 install . && \
    cd .. && \
    rm -rf flash-attention

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs ~/.huggingface

# Copy source code
COPY src/ /app/src/
COPY models/ /app/models/

# Copy and set up entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set up CUDA environment
RUN ln -s /usr/local/cuda/lib64/libcudart.so /usr/lib/libcudart.so && \
    ln -s /usr/local/cuda/lib64/libnvrtc.so /usr/lib/libnvrtc.so && \
    ldconfig

# Add CUDA libraries to path
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}

# Verify installations
RUN python3 -c "import numpy; print('NumPy version:', numpy.__version__)" && \
    python3 -c "import torch; print('PyTorch version:', torch.__version__)" && \
    python3 -c "import transformers; print('Transformers version:', transformers.__version__)" && \
    python3 -c "import bitsandbytes; print('bitsandbytes version:', bitsandbytes.__version__)"

ENTRYPOINT ["/entrypoint.sh"]
