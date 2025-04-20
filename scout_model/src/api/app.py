import os
import logging
import torch
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
import uvicorn
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/app.log')
    ]
)
logger = logging.getLogger(__name__)

# Global model interface
model_interface = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize model on startup
    global model_interface
    try:
        model_interface = ModelInterface()
        yield
    finally:
        # Cleanup on shutdown
        if model_interface and hasattr(model_interface, 'model'):
            del model_interface.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

app = FastAPI(lifespan=lifespan)

class ModelInterface:
    def __init__(self):
        try:
            # Log all environment variables (excluding token)
            env_vars = {k: v for k, v in os.environ.items() if 'TOKEN' not in k.upper()}
            logger.info(f"Environment variables: {env_vars}")

            self.model_id = os.getenv("MODEL_NAME")
            self.huggingface_token = os.getenv("HUGGING_FACE_TOKEN")
            
            if not self.model_id:
                raise ValueError("MODEL_NAME environment variable is not set")
            if not self.huggingface_token:
                raise ValueError("HUGGING_FACE_TOKEN environment variable is not set")

            logger.info(f"Initializing model interface with model: {self.model_id}")
            
            # Log in to Hugging Face
            logger.info("Logging in to Hugging Face...")
            login(token=self.huggingface_token)

            # Configure GPU and memory settings
            if torch.cuda.is_available():
                logger.info(f"CUDA available. Device count: {torch.cuda.device_count()}")
                logger.info(f"Current device: {torch.cuda.current_device()}")
                logger.info(f"Device name: {torch.cuda.get_device_name(0)}")
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                
                # Set memory fraction
                torch.cuda.set_per_process_memory_fraction(0.95)

            # Configure quantization
            logger.info("Configuring quantization...")
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            # Configure memory for L40
            max_memory = {
                0: "40GB",
                "cpu": "30GB"
            }

            # Load model
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                token=self.huggingface_token,
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16
            )

            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=self.huggingface_token
            )

            # Log success
            logger.info("Model interface initialized successfully")
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(0) / 1e9
                memory_reserved = torch.cuda.memory_reserved(0) / 1e9
                logger.info(f"GPU Memory allocated: {memory_allocated:.2f} GB")
                logger.info(f"GPU Memory reserved: {memory_reserved:.2f} GB")

        except Exception as e:
            logger.error(f"Error initializing model interface: {str(e)}", exc_info=True)
            raise

@app.get("/health")
async def health_check():
    if model_interface is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {"status": "healthy", "model": model_interface.model_id}

@app.get("/")
async def root():
    if model_interface is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    return {
        "status": "ok",
        "model": model_interface.model_id,
        "gpu_info": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB" if torch.cuda.is_available() else None
        }
    }

if __name__ == "__main__":
    try:
        port = int(os.getenv("API_PORT", 8010))
        host = os.getenv("API_HOST", "0.0.0.0")
        
        logger.info(f"Starting API server on {host}:{port}")
        uvicorn.run("src.api.app:app", host=host, port=port, reload=False, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}", exc_info=True)
        raise
