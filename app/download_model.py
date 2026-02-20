import os
import torch
from transformers import AutoTokenizer
from app.model_multitask_xlmr import SinhalaMultiHeadRegressor

MODEL_SOURCE = "akura-official/xlm-roberta-large-sinhala-multihead"

def download():
    print(f"Pre-downloading model: {MODEL_SOURCE}")
    
    # Download tokenizer
    AutoTokenizer.from_pretrained(MODEL_SOURCE, use_fast=False, trust_remote_code=True)
    print("Tokenizer downloaded.")
    
    # Download model
    # Note: We don't need to move it to GPU here as this runs during build (CPU)
    SinhalaMultiHeadRegressor.from_pretrained(MODEL_SOURCE, trust_remote_code=True)
    print("Model downloaded and cached successfully.")

if __name__ == "__main__":
    download()
