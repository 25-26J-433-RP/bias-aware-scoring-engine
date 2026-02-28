from fastapi import FastAPI, Depends, HTTPException, Header, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import uvicorn

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

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
from .fairness import FAIRNESS_PASS_CUTOFF, spd, dir_ratio, eod, binarize

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

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Bias-Aware Sinhala Essay Grader", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
# Sinhala Baseline Scoring
# -----------------------------
@app.post("/score-sinhala", dependencies=[Depends(verify_api_key)])
@limiter.limit("30/minute")
def score_sinhala(payload: SinhalaEssayIn, request: Request):
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
@limiter.limit("20/minute")
def score_sinhala_ml(payload: SinhalaEssayIn, request: Request):
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
            "topic": payload.topic,
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
@limiter.limit("20/minute")
def fairness_eval(payload: List[FairnessEvalIn], request: Request):

    scores = [p.score for p in payload]
    y_true = [p.y_true for p in payload]
    groups = [p.dyslexic_flag for p in payload]

    y_hat_bin = binarize(scores, cutoff=FAIRNESS_PASS_CUTOFF)

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
@limiter.limit("5/minute")
def trigger_fairness_analysis(background_tasks: BackgroundTasks, request: Request):
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


