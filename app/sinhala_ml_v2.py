# app/sinhala_ml_v2.py

import os
import torch
from transformers import AutoTokenizer
from .model_multitask_xlmr import SinhalaMultiHeadRegressor

MODEL_SOURCE = "akura-official/xlm-roberta-large-sinhala-multihead"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Detect CI / test environment
IS_TEST = os.getenv("DISABLE_ML", "0") == "1"

# Load model with retry logic and error handling
tokenizer = None
model = None

if not IS_TEST:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_SOURCE,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=True  # Use cached files only during startup
        )

        model = SinhalaMultiHeadRegressor.from_pretrained(
            MODEL_SOURCE,
            trust_remote_code=True,
            local_files_only=True  # Use cached files only during startup
        )
    except Exception as e:
        print(f"Warning: Could not load model from cache: {e}")
        print("Attempting to download model from HuggingFace...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_SOURCE,
                use_fast=False,
                trust_remote_code=True
            )
            model = SinhalaMultiHeadRegressor.from_pretrained(
                MODEL_SOURCE,
                trust_remote_code=True
            )
        except Exception as e2:
            print(f"Error: Failed to load model: {e2}")
            raise

    model.to(DEVICE)
    model.eval()
else:
    # CI-safe placeholders
    tokenizer = None
    model = None


def _get_grade_adjustment_factor(grade: int, text_length: int) -> float:
    """
    Calculate a grade-level adjustment factor.
    Higher grades have stricter expectations, so scores are adjusted downward
    for simpler essays when evaluated at higher grades.
    
    Grade expectations (on 14-point scale):
    - Grade 3: expects ~6-8 points (more lenient)
    - Grade 4: expects ~7-9 points
    - Grade 5: expects ~8-10 points (middle)
    - Grade 6: expects ~9-11 points
    - Grade 7: expects ~10-12 points
    - Grade 8: expects ~11-13 points (strictest)
    
    Args:
        grade: Student grade level (3-8)
        text_length: Word count of essay
        
    Returns:
        Adjustment multiplier (< 1.0 reduces score, > 1.0 increases score)
    """
    # Base adjustment by grade (stricter = lower multiplier)
    grade_multipliers = {
        3: 1.15,  # Most lenient - boosts score
        4: 1.10,
        5: 1.05,  # Near middle
        6: 0.98,
        7: 0.92,
        8: 0.85,  # Strictest - reduces score most
    }
    
    multiplier = grade_multipliers.get(grade, 1.0)
    
    # Adjust based on essay length
    # Very short essays should be penalized more at higher grades
    if text_length < 50:
        multiplier *= 0.90  # Additional 10% penalty for very short essays at any grade
    elif text_length < 100:
        multiplier *= 0.95  # 5% penalty for short essays
    
    return multiplier


def score_sinhala_ml_v2(text: str, grade: int) -> dict:
    # 🔹 CI-safe dummy output (no ML load)
    if model is None:
        # Apply grade adjustment even to dummy output
        text_length = len(text.split())
        adjustment = _get_grade_adjustment_factor(grade, text_length)
        
        base_richness = 3.0
        base_organization = 3.5
        base_technical = 2.0
        base_total = 8.5
        
        return {
            "richness_5": round(base_richness * adjustment, 2),
            "organization_6": round(base_organization * adjustment, 2),
            "technical_3": round(base_technical * adjustment, 2),
            "total_14": round(base_total * adjustment, 2),
        }

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

    # Apply grade-aware adjustment to outputs
    text_length = len(text.split())
    adjustment_factor = _get_grade_adjustment_factor(grade, text_length)
    
    richness_adjusted = float(outputs["richness_5"]) * adjustment_factor
    organization_adjusted = float(outputs["organization_6"]) * adjustment_factor
    technical_adjusted = float(outputs["technical_3"]) * adjustment_factor
    total_adjusted = float(outputs["total_14"]) * adjustment_factor

    return {
        "richness_5": round(richness_adjusted, 2),
        "organization_6": round(organization_adjusted, 2),
        "technical_3": round(technical_adjusted, 2),
        "total_14": round(total_adjusted, 2),
    }
