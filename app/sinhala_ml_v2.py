import os
import torch
from transformers import AutoTokenizer
from app.sin_model_v2 import SinhalaRegressorV2

MODEL_NAME = "xlm-roberta-base"
MODEL_PATH = "models/sinhala_v2_regressor.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔥 Detect whether we are inside GitHub Actions (CI mode)
SKIP_MODEL_LOAD = os.getenv("SKIP_MODEL_LOAD") == "true"

if SKIP_MODEL_LOAD:
    print("⚠️ SKIP_MODEL_LOAD=true → CI mode active. Sinhala ML model will NOT be loaded.")
    tokenizer = None
    model = None

else:
    print("🔄 Loading Sinhala XLM-R V2 model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = SinhalaRegressorV2(model_name=MODEL_NAME)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("✅ Sinhala XLM-R V2 model loaded.")

except Exception as e:
    print("⚠️ Model load failed → Using fallback baseline scorer.")
    print("Reason:", str(e))
    model = None



def score_sinhala_ml_v2(text: str, topic: str) -> float:
    """Predict a score using the trained Sinhala V2 model."""

    if model is None:
        raise RuntimeError("❌ Sinhala ML model disabled in CI mode. Enable it in production/local run.")

    # Combine topic + essay
    combined = f"[TOPIC={topic}] " + text

    enc = tokenizer(
        combined,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    with torch.no_grad():
        output = model(input_ids, attention_mask).squeeze().item()

    output = max(0, min(100, output))
    return round(output, 2)
