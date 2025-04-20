#!/bin/bash

# Function to check GPU availability
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found. Is the NVIDIA driver installed?"
        exit 1
    }

    if ! nvidia-smi &> /dev/null; then
        echo "ERROR: Unable to access GPU. Check NVIDIA driver and GPU status."
        exit 1
    }

    echo "GPU check passed. Found the following GPU(s):"
    nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv,noheader
}

# Function to check environment variables
check_env() {
    if [ -z "$HUGGING_FACE_TOKEN" ]; then
        echo "ERROR: HUGGING_FACE_TOKEN is not set"
        exit 1
    fi
}

# Function to verify CUDA setup
verify_cuda() {
    python3 -c "
import torch
import bitsandbytes as bnb
print('CUDA available:', torch.cuda.is_available())
print('CUDA version:', torch.version.cuda)
print('bitsandbytes version:', bnb.__version__)
"
}

# Main
echo "Starting container initialization..."

# Check GPU
echo "Checking GPU availability..."
check_gpu

# Check environment variables
echo "Checking environment variables..."
check_env

# Verify CUDA setup
echo "Verifying CUDA setup..."
verify_cuda

# Print Python version and packages
echo "Python environment:"
python3 --version
pip3 list

# Execute the main command
echo "Starting application..."
exec "$@"
