import os
from pathlib import Path

# Project Directory Structure
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FASTA_DIR = DATA_DIR / "fasta_files"
SPIKE_DIR = DATA_DIR / "spike_sequences"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-4-Scout-17B-16E")
MAX_TOKEN_LENGTH = int(os.getenv("MAX_TOKEN_LENGTH", 35000))

# Spike Protein Configuration
SPIKE_COORDINATES = {
    "START": 21563,  # Start position of spike protein in genome
    "END": 25384     # End position of spike protein in genome
}

# Training Configuration
TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-4,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03
}

# LoRA Configuration
LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
}

# Quantization Configuration
QUANTIZATION_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_use_double_quant": True
}

# Data Processing
SEQUENCE_TEMPLATES = {
    "instruction": "Based on this COVID-19 spike protein sequence, which country does this sample come from?",
    "response_template": "This COVID-19 sample originates from {country}. This classification is based on the genomic characteristics of the spike protein sequence."
}

# File Extensions
VALID_EXTENSIONS = [".fasta", ".fa", ".fna"]

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "training.log"),
            "formatter": "standard"
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard"
        }
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    }
}


# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": int(os.getenv("API_PORT", 8000)),
    "reload": False,
    "workers": 1
}

# Add a function to get the actual port
def get_api_port():
    """Get the actual port being used by the container"""
    return int(os.getenv("PORT", API_CONFIG["port"]))
    
# Model Generation Parameters
GENERATION_CONFIG = {
    "max_new_tokens": 100,
    "temperature": 0.7,
    "num_return_sequences": 1,
    "do_sample": True,
    "top_p": 0.95,
    "top_k": 50
}


# Error Messages
ERROR_MESSAGES = {
    "file_not_found": "FASTA file not found: {}",
    "invalid_sequence": "Invalid sequence format in file: {}",
    "model_load_error": "Error loading model: {}",
    "processing_error": "Error processing sequence: {}",
    "token_length_exceeded": "Sequence token length ({}) exceeds maximum allowed ({})"
}

# Validation
VALIDATION = {
    "min_sequence_length": 100,  # Minimum acceptable sequence length
    "max_sequence_length": 10000,  # Maximum acceptable sequence length
    "allowed_nucleotides": set('ATCGN-'),  # Valid nucleotides in sequence
    "min_quality_score": 20  # Minimum acceptable quality score
}

# Cache Configuration
CACHE_CONFIG = {
    "max_size": 1000,  # Maximum number of items in cache
    "ttl": 3600  # Time to live in seconds
}

# Model Checkpoint Configuration
CHECKPOINT_CONFIG = {
    "save_steps": 50,
    "save_total_limit": 3,
    "save_safetensors": True
}

# Environment Variables (with defaults)
ENV_VARS = {
    "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", "0"),
    "TORCH_CUDA_ARCH_LIST": os.getenv("TORCH_CUDA_ARCH_LIST", "7.5"),
    "TRANSFORMERS_CACHE": str(PROJECT_ROOT / "cache"),
    "HF_HOME": str(PROJECT_ROOT / "cache/huggingface")
}

# Set environment variables
for key, value in ENV_VARS.items():
    os.environ[key] = str(value)
