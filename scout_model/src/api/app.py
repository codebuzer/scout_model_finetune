import os
import logging
import torch
import gc
import math
import psutil
import numpy as np
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import uvicorn
from pydantic import BaseModel
from torch.cuda.amp import autocast
from contextlib import nullcontext
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils import prune
import torch.nn.utils.prune as prune_utils
from datetime import datetime
import json
 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Model compression configuration
PRUNING_CONFIG = {
    'structured_amount': 0.2,    # Remove 20% of channels/neurons
    'unstructured_amount': 0.3,  # Remove 30% of individual weights
    'min_channels': 16,          # Minimum number of channels to keep
}

# Quantization Configuration
QUANTIZATION_CONFIG = {
    'ternary': {
        'threshold': 0.7,        # Threshold for ternary quantization
        'chunk_size': 1024,       # Chunk size for memory-efficient processing
    },
    'activation': {
        'bits': 8,              # 8-bit activation quantization
        'scale_factor': 127,    # Scale factor for 8-bit quantization (2^7 - 1)
    },
    'memory': {
        'gpu_memory_fraction': 0.9,  # Maximum GPU memory fraction to use
        'cpu_offload_threshold': 0.8  # When to offload to CPU
    }
}


# ===============================
# Utilities
# ===============================


def calculate_sparsity(model):
    """
    Calculate sparsity with memory-efficient chunked processing
    """
    with torch.no_grad():
        total_params = 0
        zero_params = 0
        chunk_size = 1000000  # Process 1M elements at a time
        
        # Get weight tensors
        weight_tensors = [p for n, p in model.named_parameters() if 'weight' in n]
        
        for tensor in weight_tensors:
            # Add to total params
            total_params += tensor.numel()
            
            # Process in chunks
            for i in range(0, tensor.numel(), chunk_size):
                # Get chunk
                end_idx = min(i + chunk_size, tensor.numel())
                chunk = tensor.view(-1)[i:end_idx]
                
                # Move to CPU for counting if needed
                if chunk.device.type == 'cuda':
                    chunk = chunk.cpu()
                
                # Count zeros
                zero_params += (chunk == 0).sum().item()
                
                # Clear memory
                del chunk
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return zero_params / total_params if total_params > 0 else 0

def structured_pruning(module, name='weight', amount=0.3):
    """Apply structured pruning to a module"""
    if hasattr(module, name):
        prune.ln_structured(
            module,
            name=name,
            amount=amount,
            n=2,  # L2 norm
            dim=0  # Prune entire output channels
        )

def unstructured_pruning(module, name='weight', amount=0.3):
    """Apply unstructured pruning to a module"""
    if hasattr(module, name):
        prune.l1_unstructured(
            module,
            name=name,
            amount=amount
        )
        
def batch_structured_pruning(layers, amount):
    """
    Structured pruning with memory efficiency
    """
    for layer in tqdm(layers, desc="Structured pruning"):
        if hasattr(layer, 'weight'):
            try:
                # Get weight data and ensure it's materialized
                weight_data = layer.weight.data
                if weight_data.device.type == 'meta':
                    continue  # Skip meta tensors
                
                # Move to CPU for processing if needed
                original_device = weight_data.device
                weight_data = weight_data.cpu()
                
                # Calculate L2 norms
                norms = torch.norm(weight_data, p=2, dim=1)
                
                # Use sorting instead of kthvalue
                num_channels = int(amount * len(norms))
                sorted_norms, indices = torch.sort(norms)
                threshold = sorted_norms[num_channels]
                
                # Create and apply mask
                mask = (norms > threshold).float().view(-1, 1)
                layer.weight.data = (weight_data * mask).to(original_device)
                
                # Clear memory
                del weight_data, norms, mask, sorted_norms, indices
                clear_memory()
                
            except Exception as e:
                logger.warning(f"Could not prune layer: {str(e)}")
                continue

        
def prune_model(model, structured_amount=0.2, unstructured_amount=0.3):
    """
    Apply both structured and unstructured pruning to the model
    Args:
        model: The model to prune
        structured_amount: Amount of structured pruning (0-1)
        unstructured_amount: Amount of unstructured pruning (0-1)
    Returns:
        Pruned model
    """
    try:
        logger.info("Starting model pruning...")
        
        # Calculate initial metrics
        initial_size = sum(p.numel() for p in model.parameters() if p.device.type != 'meta')
        initial_sparsity = calculate_sparsity(model)
        
        # Get prunable layers
        prunable_layers = [
            m for m in model.modules()
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d))
            and hasattr(m, 'weight')
            and m.weight.device.type != 'meta'
        ]
        
        if not prunable_layers:
            logger.warning("No prunable layers found, skipping pruning")
            return model
            
        # Apply structured pruning
        logger.info("Applying structured pruning...")
        batch_structured_pruning(prunable_layers, structured_amount)
        clear_memory(full_clean=True)
        
        # Apply unstructured pruning
        logger.info("Applying unstructured pruning...")
        batch_unstructured_pruning(prunable_layers, unstructured_amount)
        clear_memory(full_clean=True)
        
        # Calculate final metrics
        final_sparsity = calculate_sparsity(model)
        final_size = sum(p.numel() for p in model.parameters() if p.device.type != 'meta')
        
        # Log results
        logger.info("Pruning complete:")
        logger.info(f"Initial model size: {initial_size:,} parameters")
        logger.info(f"Final model size: {final_size:,} parameters")
        logger.info(f"Initial sparsity: {initial_sparsity:.2%}")
        logger.info(f"Final sparsity: {final_sparsity:.2%}")
        logger.info(f"Size reduction: {(1 - final_size/initial_size):.2%}")
        
        return model

    except Exception as e:
        logger.error(f"Error during pruning: {str(e)}")
        raise




def compress_model(model):
    """
    Apply comprehensive model compression (pruning + quantization)
    """
    try:
        # Record initial metrics
        initial_params = sum(p.numel() for p in model.parameters() if p.device.type != 'meta')
        initial_sparsity = calculate_sparsity(model)
        
        logger.info(f"Initial model stats:")
        logger.info(f"Parameters: {initial_params:,}")
        logger.info(f"Sparsity: {initial_sparsity:.2%}")
        
        # Step 1: Pruning
        logger.info("Starting model compression...")
        try:
            model = prune_model(model)
        except Exception as e:
            logger.warning(f"Pruning failed, continuing with quantization: {str(e)}")
        
        clear_memory(full_clean=True)
        
        # Step 2: Quantization
        logger.info("Applying quantization...")
        model = quantize_model(model)
        clear_memory(full_clean=True)
        
        # Calculate final metrics
        final_params = sum(p.numel() for p in model.parameters() if p.device.type != 'meta')
        final_sparsity = calculate_sparsity(model)
        
        # Log comprehensive results
        logger.info(f"Compression complete:")
        logger.info(f"Initial parameters: {initial_params:,}")
        logger.info(f"Final parameters: {final_params:,}")
        logger.info(f"Initial sparsity: {initial_sparsity:.2%}")
        logger.info(f"Final sparsity: {final_sparsity:.2%}")
        logger.info(f"Parameter reduction: {(1 - final_params/initial_params):.2%}")
        logger.info(f"Sparsity increase: {(final_sparsity - initial_sparsity):.2%}")
        
        return model
    
    except Exception as e:
        logger.error(f"Error during model compression: {str(e)}")
        raise
def batch_unstructured_pruning(layers, amount):
    """
    Apply unstructured pruning to layers in batches
    Args:
        layers: List of model layers to prune
        amount: Amount of weights to prune (0-1)
    """
    try:
        for layer in tqdm(layers, desc="Unstructured pruning"):
            if hasattr(layer, 'weight'):
                # Skip meta tensors
                if layer.weight.device.type == 'meta':
                    continue
                    
                # Get weight data and device
                weight_data = layer.weight.data
                original_device = weight_data.device
                
                # Move to CPU if on CUDA
                if weight_data.device.type == 'cuda':
                    weight_data = weight_data.cpu()
                
                try:
                    # Calculate threshold and create mask
                    threshold = torch.quantile(weight_data.abs().float(), amount)
                    mask = (weight_data.abs() > threshold).float()
                    
                    # Apply mask and move back to original device
                    layer.weight.data = (weight_data * mask).to(original_device)
                    
                    # Clear intermediate tensors
                    del weight_data, mask, threshold
                    clear_memory()
                    
                except Exception as e:
                    logger.warning(f"Error pruning layer: {str(e)}")
                    continue
                    
    except Exception as e:
        logger.error(f"Error in batch unstructured pruning: {str(e)}")
        raise

def set_memory_limits():
    """Configure system memory limits and PyTorch settings"""
    if torch.cuda.is_available():
        # More conservative memory settings
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use less GPU memory
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
            'max_split_size_mb:128,'
            'garbage_collection_threshold:0.6,'
            'roundup_power2:True,'
            'allocation_timeout:30'
        )
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # CPU Memory Limits
    memory = psutil.virtual_memory()
    cpu_limit = int(memory.total * 0.7)  # Use less system RAM
    torch.set_num_threads(max(1, psutil.cpu_count() // 4))  # Use fewer CPU threads

    
def log_memory_usage():
    """Detailed memory usage logging"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1e9
            memory_reserved = torch.cuda.memory_reserved(i) / 1e9
            memory_cached = torch.cuda.memory_cached(i) / 1e9
            logger.info(f"GPU {i} Memory (GB) - Allocated: {memory_allocated:.2f}, Reserved: {memory_reserved:.2f}, Cached: {memory_cached:.2f}")
    
    process = psutil.Process()
    cpu_memory = process.memory_info().rss / 1e9
    logger.info(f"CPU Memory Usage: {cpu_memory:.2f} GB")  

def clear_memory(full_clean=False):
    """Advanced memory cleanup"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    if full_clean:
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    del obj
            except Exception:
                pass
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Ternary quantization for weights: {-1, 0, 1}
def ternarize_weights(tensor, config=QUANTIZATION_CONFIG['ternary']):
    
    with torch.no_grad():
        if tensor.device.type == 'meta':
            return tensor, 1.0

        try:
            tensor = tensor.cpu().float()
            scale = tensor.abs().mean()
            normalized = tensor / (scale + 1e-5)

            ternary_weights = torch.zeros_like(normalized)
            ternary_weights = torch.where(normalized > config['threshold'], torch.ones_like(normalized),
                              torch.where(normalized < -config['threshold'], -torch.ones_like(normalized),
                              torch.zeros_like(normalized)))
        
            return ternary_weights, scale
  
        except Exception as e:
            logger.warning(f"Could not quantize tensor: {e}")
            return tensor, 1.0
            
            
def optimize_model_memory(model):
    """Apply various memory optimization techniques"""
    for module in model.modules():
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.requires_grad_(False)  # Disable gradients
        if hasattr(module, 'bias') and module.bias is not None:
            module.bias.requires_grad_(False)
    
    # Enable gradient checkpointing
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    return model

# Activation quantization hook (simulate 8-bit quantization)
def quantize_activations_hook(module, input, output):
    """8-bit activation quantization with caching"""
    config = QUANTIZATION_CONFIG['activation']
    with torch.no_grad():
        if hasattr(module, '_activation_range'):
            abs_max = module._activation_range
        else:
            abs_max = output.abs().max().item()
            module._activation_range = abs_max
        
        scale = config['scale_factor'] / (abs_max + 1e-5)
        return torch.round(output * scale) / scale

def quantize_model(model):
    """
    Comprehensive model quantization with memory optimizations
    """
    try:
        with torch.no_grad():
            # First pass: collect statistics
            param_stats = {}
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    # Skip meta tensors in statistics
                    if param.device.type != 'meta':
                        param_stats[name] = {
                            'shape': param.shape,
                            'size': param.numel(),
                            'device': param.device
                        }
            
            # Sort parameters by size for optimal processing
            sorted_params = sorted(param_stats.items(), key=lambda x: x[1]['size'], reverse=True)
            
            # Second pass: quantize
            memory_config = QUANTIZATION_CONFIG['memory']
            for name, stats in tqdm(sorted_params, desc='Quantizing parameters'):
                clear_memory()
                param = model.get_parameter(name)
                
                # Skip meta tensors
                if param.device.type == 'meta':
                    logger.info(f"Skipping meta tensor: {name}")
                    continue
                
                # Process on CPU if GPU memory is tight
                if (torch.cuda.is_available() and 
                    torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory > 
                    memory_config['cpu_offload_threshold']):  # Fixed the parentheses here
                    try:
                        param.data = param.cpu()
                    except Exception as e:
                        logger.warning(f"Could not move {name} to CPU: {e}")
                        continue

                logger.info(f"[Quantization] Processing {name} with shape {param.shape} and size {param.numel()}")

                try:
                    # Quantize
                    ternary_weights, scale = ternarize_weights(param.data)
                    param.data = (ternary_weights * scale).to(param.device)
                except Exception as e:
                    logger.warning(f"Could not quantize {name}: {e}")
                    continue
                
                clear_memory()
            
            # Optimize model memory usage
            model = optimize_model_memory(model)
            
        return model
    except Exception as e:
        logger.error(f"Error during quantization: {str(e)}")
        raise


# Apply activation hooks to Linear layers
def add_activation_hooks(model, config=QUANTIZATION_CONFIG['activation']):
    """
    Optimized activation hook addition
    """
    hooks = []
    try:
        logger.info("Adding activation quantization hooks...")
        
        # Vectorized hook function
        def quantization_hook(module, input, output):
            with torch.no_grad():
                if not hasattr(module, '_activation_range'):
                    module._activation_range = output.abs().max().item()
                
                # Vectorized quantization
                scale = config['scale_factor'] / (module._activation_range + 1e-5)
                return torch.round(output * scale) / scale
        
        # Batch process layers
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        hooks.extend([
            module.register_forward_hook(quantization_hook)
            for module in tqdm(linear_layers, desc="Adding activation hooks")
        ])
        
        logger.info(f"Added quantization hooks to {len(hooks)} layers")
        return hooks
    
    except Exception as e:
        logger.error(f"Error adding activation hooks: {str(e)}")
        for hook in hooks:
            hook.remove()
        raise

def load_model_parameters(model):
    """
    Optimized parameter loading
    """
    try:
        # Batch process parameters
        meta_params = {
            name: param for name, param in model.named_parameters()
            if param.device.type == 'meta'
        }
        
        # Process in batches to manage memory
        batch_size = 10
        for i in range(0, len(meta_params), batch_size):
            batch = list(meta_params.items())[i:i+batch_size]
            for name, param in tqdm(batch, desc=f"Loading parameters batch {i//batch_size + 1}"):
                try:
                    param.data = param.data.to(param.device)
                except Exception as e:
                    logger.warning(f"Could not materialize parameter {name}: {e}")
            
            clear_memory()
            
    except Exception as e:
        logger.error(f"Error loading parameters: {str(e)}")
        raise

# ===============================
# Model Interface
# ===============================

class ModelInterface:
    def __init__(self):
        try:
            set_memory_limits()
            clear_memory(full_clean=True)
            
            self.model_id = os.getenv("MODEL_NAME", "meta-llama/Llama-4-Scout-17B-16E-Instruct")
            self.huggingface_token = os.getenv("HUGGING_FACE_TOKEN")

            if not self.huggingface_token:
                raise ValueError("HUGGING_FACE_TOKEN environment variable is not set")

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")

            logger.info(f"Initializing model: {self.model_id}")
            login(token=self.huggingface_token)
            

            # Optimize memory settings
            n_gpus = torch.cuda.device_count()
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            max_memory = {
                i: f"{int(gpu_memory * 0.7)}GB" for i in range(n_gpus)
            }
            max_memory["cpu"] = f"{int(psutil.virtual_memory().total / 1e9 * 0.7)}GB"


            # Load tokenizer first
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=self.huggingface_token,
                use_fast=True  # Use fast tokenizer
            )
            

            # Load model with optimizations
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                token=self.huggingface_token,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                offload_folder="offload",
                offload_state_dict=True
            )
            logger.info("Waiting for model to be fully loaded...")
            self.model.eval()  # Ensure model is in eval mode
            
            # Force load parameters from disk
            logger.info("Loading parameters from disk...")
            load_model_parameters(self.model)
            
            # Apply optimizations
            logger.info("Starting model compression pipeline...")
            clear_memory(full_clean=True)
            log_memory_usage()

            logger.info("Applying compression...")
            try:
                self.model = compress_model(self.model)
            except Exception as e:
                logger.warning(f"Compression failed, continuing with original model: {str(e)}")
            
            # Apply activation quantization
            logger.info("Applying activation quantization...")
            self.activation_hooks = add_activation_hooks(self.model)
            

            # Final memory optimization
            clear_memory(full_clean=True)
            log_memory_usage()
            logger.info("Model initialization and compression completed successfully!")

        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}", exc_info=True)
            raise
            
            
    def __del__(self):
        """
        Cleanup when the interface is destroyed
        """
        try:
            # Remove activation hooks to prevent memory leaks
            if hasattr(self, 'activation_hooks'):
                for hook in self.activation_hooks:
                    hook.remove()
        except Exception as e:
            logger.error(f"Error cleaning up activation hooks: {str(e)}")
      
    
    async def generate_response(self, prompt, max_length=2048, temperature=0.7):
        try:
            clear_memory()

            # Optimize input processing
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate with memory optimization
            with torch.no_grad(), autocast(enabled=True):
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    use_cache=True,
                    repetition_penalty=1.1,
                    length_penalty=1.0
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            clear_memory()
            
            return response

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

# ===============================
# FastAPI Setup
# ===============================

app = FastAPI()
model_interface = None

@app.on_event("startup")
async def startup_event():
    global model_interface
    model_interface = ModelInterface()

@app.on_event("shutdown")
async def shutdown_event():
    global model_interface
    if model_interface is not None:
        clear_memory(full_clean=True)
        model_interface = None

@app.get("/health")
async def health_check():
    if model_interface is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {
        "status": "healthy",
        "model": model_interface.model_id,
        "gpu_memory": {f"gpu_{i}": f"{torch.cuda.memory_allocated(i)/1e9:.2f}GB" 
                      for i in range(torch.cuda.device_count())}
    }

class GenerateRequest(BaseModel):
    prompt: str
    max_length: int = 2048
    temperature: float = 0.7

@app.post("/generate")
async def generate(request: GenerateRequest):
    if model_interface is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    try:
        clear_memory()
        response = await model_interface.generate_response(
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===============================
# Logger
# ===============================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/app.log')
    ]
)
logger = logging.getLogger(__name__)

# ===============================
# Main Entry
# ===============================

if __name__ == "__main__":
    try:
        set_memory_limits()
        port = int(os.getenv("API_PORT", 8010))
        host = os.getenv("API_HOST", "0.0.0.0")
        
        logger.info(f"Starting API server on {host}:{port}")
        uvicorn.run(
            "src.api.app:app", 
            host=host, 
            port=port, 
            reload=False, 
            log_level="info",
            workers=1,
            limit_concurrency=1,
            timeout_keep_alive=30
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        raise
