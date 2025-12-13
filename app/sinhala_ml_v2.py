# app/sinhala_ml_v2.py

import torch
from transformers import AutoTokenizer
from .sin_model_v2 import SinhalaRegressorV2

BASE_TOKENIZER = "xlm-roberta-large"
MODEL_NAME = "akura-official/xlm-roberta-large-sinhala-multihead"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔄 Loading Sinhala multi-head model...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_TOKENIZER,
    use_fast=False
)

model = SinhalaRegressorV2(MODEL_NAME)

# 🔑 🔑 🔑 THIS IS THE FIX
model.encoder.resize_token_embeddings(len(tokenizer))

model.to(DEVICE)
model.eval()

print("✅ Sinhala ML model loaded successfully.")


def score_sinhala_ml_v2(text: str, grade: int, topic: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    return {
        "richness_5": round(outputs["richness_5"].item(), 2),
        "organization_6": round(outputs["organization_6"].item(), 2),
        "technical_3": round(outputs["technical_3"].item(), 2),
        "total_14": round(outputs["total_14"].item(), 2),
    }
