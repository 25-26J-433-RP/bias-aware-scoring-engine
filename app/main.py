from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from .schemas import (
    EssayIn, ScoreOut,
    SinhalaEssayIn, SinhalaMLOut,
    SinhalaRubricOut, FairnessReport,
    FairnessEvalIn
)

from .scorer import score_essay
from .sinhala_baseline import baseline_sinhala_score
from .sinhala_ml_v2 import score_sinhala_ml_v2
from .grade_detector import infer_grade_from_text
from .fairness import spd, dir_ratio, eod, binarize

app = FastAPI(title="Bias-Aware Sinhala Essay Grader", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    print(f"[DEBUG] Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    print(f"[DEBUG] Request completed: {request.method} {request.url} -> {response.status_code}")
    return response


# -----------------------------
# Health
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# English Essay Scoring
# -----------------------------
@app.post("/score", response_model=ScoreOut)
def score(payload: EssayIn):
    score_value, details = score_essay(payload.text, payload.prompt)
    return {
        "score": round(score_value, 2),
        "details": details,
        "fairness_report": None
    }


# -----------------------------
# Sinhala Baseline Scoring
# -----------------------------
@app.post("/score-sinhala")
def score_sinhala(payload: SinhalaEssayIn):
    score, details = baseline_sinhala_score(payload.text)
    return {
        "score": score,
        "details": details,
        "rubric": SinhalaRubricOut().model_dump(),
        "fairness_report": None
    }


# -----------------------------
# Sinhala ML Scoring (MAIN)
# Grade-aware: Automatically detects grade if not provided
# -----------------------------
@app.post("/score-sinhala-ml", response_model=SinhalaMLOut)
def score_sinhala_ml(payload: SinhalaEssayIn):
    print(f"--- [REQUEST] Scoring Sinhala Essay (Grade {payload.grade}) ---")
    # Detect or use provided grade
    detected_grade = infer_grade_from_text(payload.text, payload.grade)
    
    # DEBUG: Log what grade was received and used
    print(f"[SCORE-SINHALA-ML] Received grade: {payload.grade}, Using grade: {detected_grade}")
    
    scores = score_sinhala_ml_v2(
        text=payload.text,
        grade=detected_grade,
        dyslexic_flag=payload.dyslexic_flag
    )

    final_score = min(100, (scores["total_14"] / 14) * 100)

    return {
        "score": round(final_score, 2),
        "rubric": scores,
        "details": {
            "dyslexic_flag": payload.dyslexic_flag,
            "error_tags": payload.error_tags,
            "model": "✅ RETRAINED MODEL (Cloud)",
            "detected_grade": detected_grade,
            "grade_auto_detected": payload.grade is None
        },
        "fairness_report": {
            "spd": scores.get("mitigation_info", {}).get("fairness_metrics", {}).get("spd", 0.0) if "mitigation_info" in scores else 0.0,
            "dir": scores.get("mitigation_info", {}).get("fairness_metrics", {}).get("dir", 1.0) if "mitigation_info" in scores else 1.0,
            "mitigation_used": scores.get("mitigation_info", None)
        } if "mitigation_info" in scores else None
    }

# -----------------------------
# Batch Fairness Evaluation
# -----------------------------
@app.post("/fairness-eval", response_model=FairnessReport)
def fairness_eval(payload: List[FairnessEvalIn]):

    scores = [p.score for p in payload]
    y_true = [p.y_true for p in payload]
    groups = [p.dyslexic_flag for p in payload]

    y_hat_bin = binarize(scores)

    return FairnessReport(
        spd=spd(y_hat_bin, groups),
        dir=dir_ratio(y_hat_bin, groups),
        eod=eod(y_hat_bin, y_true, groups),
        mitigation_used="Reweighing (planned)"
    )


# -----------------------------
# Cloud Run / Server startup
# -----------------------------
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # Use 0.0.0.0 only in production (Cloud Run/Docker) for container networking
    # Default to 127.0.0.1 for local development (more secure)
    host = os.environ.get("HOST", "127.0.0.1")
    uvicorn.run(
        app,
        host=host,  # noqa: S104 - Required for Cloud Run deployment
        port=port,
        log_level="info"
    )

