# app/sinhala_ml_v2.py

import torch
from transformers import AutoTokenizer
from .model_multitask_xlmr import SinhalaMultiHeadRegressor

MODEL_SOURCE = "models/xlm-roberta-large-sinhala-multihead"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE, use_fast=False)
model = SinhalaMultiHeadRegressor.from_pretrained(MODEL_SOURCE)
model.to(DEVICE)
model.eval()

def score_sinhala_ml_v2(text: str, grade: int) -> dict:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # ✅ MUST be torch.long
    grade_tensor = torch.tensor([grade], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            grade_id=grade_tensor
        )

    return {
        "richness_5": round(float(outputs["richness_5"]), 2),
        "organization_6": round(float(outputs["organization_6"]), 2),
        "technical_3": round(float(outputs["technical_3"]), 2),
        "total_14": round(float(outputs["total_14"]), 2),
    }
