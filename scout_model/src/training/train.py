import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer
import os

def setup_model_and_tokenizer():
    # Configure model for extended context length
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load model with extended context length
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-4-Scout-17B-16E",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        max_position_embeddings=35000,  # Extended context length
        use_flash_attention_2=True      # Enable Flash Attention 2 for better memory efficiency
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-4-Scout-17B-16E",
        trust_remote_code=True,
        model_max_length=35000
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def prepare_model_for_training(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads()
    
    return model, lora_config

def train_model():
    # First process the spike sequences
    os.system("python3 extract_spike.py")
    os.system("python3 preprocess.py")
    
    model, tokenizer = setup_model_and_tokenizer()
    model, lora_config = prepare_model_for_training(model)
    
    dataset = load_dataset('json', data_files='processed_dataset.json')
    
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=1,  # Reduced batch size due to longer sequences
        gradient_accumulation_steps=16,  # Increased for effective batch size
        save_steps=50,
        logging_steps=10,
        learning_rate=2e-4,
        fp16=True,
        optim="paged_adamw_32bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        gradient_checkpointing=True,
        # Flash Attention specific settings
        use_flash_attention=True,
        use_flash_attn_2=True
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=training_args,
        tokenizer=tokenizer,
        peft_config=lora_config,
        max_seq_length=35000
    )
    
    trainer.train()
    trainer.save_model("./final_model")

if __name__ == "__main__":
    train_model()
