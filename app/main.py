from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    EssayIn, ScoreOut,
    SinhalaEssayIn, SinhalaMLIn, SinhalaMLOut,
    FairnessReport, SinhalaRubricOut
)

from .scorer import score_essay
from .fairness import spd, dir_ratio, eod, binarize    # FAIRNESS FUNCTIONS
from .sinhala_baseline import baseline_sinhala_score
from app.sinhala_ml_v2 import score_sinhala_ml_v2


app = FastAPI(title="Bias-Aware Scoring Engine", version="0.5.0")


# ------------------------------
#  CORS
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://192.168.1.5:8081",
        "http://localhost:19006",
        "http://127.0.0.1:19006",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#  HEALTH
# ============================================================
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


# ============================================================
#  FAIRNESS HELPER
# ============================================================
def compute_fairness(score_value: float, dyslexic_flag: bool) -> FairnessReport:
    """Compute SPD, DIR, EOD for a SINGLE score entry."""

    groups = [dyslexic_flag]
    y_hat_bin = binarize([score_value], cutoff=60)
    y_true = [1]   # Placeholder

    return FairnessReport(
        spd=spd(y_hat_bin, groups),
        dir=dir_ratio(y_hat_bin, groups),
        eod=eod(y_hat_bin, y_true, groups),
        mitigation_used="None (Training Pending)"
    )


# ============================================================
#  ENGLISH ESSAY SCORING
# ============================================================
@app.post("/score", response_model=ScoreOut)
def score(payload: EssayIn):
    score_value, details = score_essay(payload.text, payload.prompt)
    fairness = compute_fairness(score_value, False)
    return {
        "score": round(score_value, 2),
        "details": details,
        "fairness_report": fairness.model_dump()
    }


# ============================================================
#  PHASE 5 — UPDATED SINHALA BASELINE ENDPOINT
# ============================================================
@app.post("/score-sinhala", tags=["sinhala"])
def score_sinhala(payload: SinhalaEssayIn):
    """
    Sinhala baseline scoring + fairness + full rubric structure.
    """

    # Baseline scoring (0–100)
    score_value, base_details = baseline_sinhala_score(payload.text)

    fairness = compute_fairness(score_value, payload.dyslexic_flag)

    source_type = "reconstructed" if payload.dyslexic_flag else "original"

    return {
        "score": score_value,
        "details": {
            "model": "baseline_rule_based",
            "grade": payload.grade,
            "topic": payload.topic,
            "dyslexic_flag": payload.dyslexic_flag,
            "error_tags": payload.error_tags,
            "source": source_type,
            "baseline_features": base_details,
        },
        "rubric": SinhalaRubricOut(
            richness_5=None,
            organization_6=None,
            technical_3=None,
            total_14=None
        ).model_dump(),
        "fairness_report": fairness.model_dump()
    }


# ============================================================
#  PHASE 5 — UPDATED SINHALA ML ENDPOINT (with fallback)
# ============================================================
@app.post("/score-sinhala-ml", tags=["sinhala"], response_model=SinhalaMLOut)
def score_sinhala_ml_route(payload: SinhalaEssayIn):
    """
    Sinhala ML scoring + fairness + full rubric output.
    Fallback to baseline if ML model is missing.
    """

    try:
        ml_score = score_sinhala_ml_v2(payload.text, payload.topic)
        model_used = "xlm-roberta-v2"
    except Exception:
        ml_score, _ = baseline_sinhala_score(payload.text)
        model_used = "baseline_fallback"

    fairness = compute_fairness(ml_score, payload.dyslexic_flag)
    source_type = "reconstructed" if payload.dyslexic_flag else "original"

    return {
        "score": ml_score,
        "details": {
            "model": model_used,
            "grade": payload.grade,
            "topic": payload.topic,
            "dyslexic_flag": payload.dyslexic_flag,
            "error_tags": payload.error_tags,
            "source": source_type,
        },
        "rubric": SinhalaRubricOut(
            richness_5=None,
            organization_6=None,
            technical_3=None,
            total_14=None
        ).model_dump(),
        "fairness_report": fairness.model_dump()
    }
