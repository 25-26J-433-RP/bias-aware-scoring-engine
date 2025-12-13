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
from .fairness import spd, dir_ratio, eod, binarize

app = FastAPI(title="Bias-Aware Sinhala Essay Grader", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
# -----------------------------
@app.post("/score-sinhala-ml", response_model=SinhalaMLOut)
def score_sinhala_ml(payload: SinhalaEssayIn):

    scores = score_sinhala_ml_v2(
        payload.text,
        payload.grade,
        payload.topic or ""
    )

    final_score = min(100, (scores["total_14"] / 14) * 100)

    return {
        "score": round(final_score, 2),
        "details": {
            "model": "xlm-roberta-multihead",
            "grade": payload.grade,
            "topic": payload.topic,
            "dyslexic_flag": payload.dyslexic_flag,
            "error_tags": payload.error_tags
        },
        "rubric": SinhalaRubricOut(**scores).model_dump(),
        "fairness_report": None
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
