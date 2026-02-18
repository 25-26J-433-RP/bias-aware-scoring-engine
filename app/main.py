from fastapi import FastAPI, Depends, HTTPException, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os

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

# -----------------------------
# Security Configuration
# -----------------------------
API_KEY_NAME = "X-API-KEY"
# In production, this should be set in environment variables
API_KEY_SECRET = os.getenv("INTERNAL_API_KEY", "akura-research-secret-2026")

async def verify_api_key(x_api_key: str = Header(..., alias=API_KEY_NAME)):
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(
            status_code=403, 
            detail="Forbidden: Invalid API Key. This endpoint is restricted."
        )
    return x_api_key

app = FastAPI(title="Bias-Aware Sinhala Essay Grader", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://akura.vercel.app",
        "https://akura-qa.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
        "http://localhost:8888",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NOTE: Model loads lazily on first request (see sinhala_ml_v2.py)
# This is the correct pattern for Cloud Run to avoid startup timeout


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
@app.post("/score-sinhala", dependencies=[Depends(verify_api_key)])
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
@app.post("/score-sinhala-ml", response_model=SinhalaMLOut, dependencies=[Depends(verify_api_key)])
def score_sinhala_ml(payload: SinhalaEssayIn):
    # Detect or use provided grade
    detected_grade = infer_grade_from_text(payload.text, payload.grade)
    
    # DEBUG: Log what grade was received and used
    print(f"[SCORE-SINHALA-ML] Received grade: {payload.grade}, Using grade: {detected_grade}")
    
    scores = score_sinhala_ml_v2(
        text=payload.text,
        grade=detected_grade,
        dyslexic_flag=payload.dyslexic_flag,
        topic=payload.topic
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
        "fairness_report": scores.get("fairness_report") if isinstance(scores, dict) else None
    }

# -----------------------------
# Batch Fairness Evaluation
# -----------------------------
@app.post("/fairness-eval", response_model=FairnessReport, dependencies=[Depends(verify_api_key)])
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
# Trigger Fairness Analysis (Research Admin)
# -----------------------------
@app.post("/run-analysis", dependencies=[Depends(verify_api_key)])
def trigger_fairness_analysis(background_tasks: BackgroundTasks):
    """
    Manually triggers the Firestore fairness evaluation script.
    Uses BackgroundTasks to prevent gateway timeouts (502 errors).
    """
    def run_analysis_task():
        try:
            # Lazy import to prevent crashes in environments without research dependencies/creds
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from analysis.firestore_fairness_eval import run_fairness_eval
            
            print("[BACKGROUND] Starting fairness analysis...")
            run_fairness_eval()
            print("[BACKGROUND] Fairness analysis completed successfully.")
        except Exception as e:
            print(f"[BACKGROUND ERROR] Error running analysis: {e}")

    # Add to background tasks and return immediately
    background_tasks.add_task(run_analysis_task)
    
    return {
        "status": "success", 
        "message": "Fairness analysis has been started in the background. Results will appear in Firestore shortly."
    }


# -----------------------------
# Cloud Run / Server startup
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
