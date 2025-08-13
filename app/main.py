from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from .schemas import EssayIn, ScoreOut
from .scorer import score_essay
from .fairness import empty_fairness_report

app = FastAPI(title="Bias-Aware Scoring Engine", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

@app.post("/score", response_model=ScoreOut)
def score(payload: EssayIn):
    score_value, details = score_essay(payload.text, payload.prompt)
    fr = empty_fairness_report()
    return {"score": round(score_value, 2), "details": details, "fairness_report": fr.model_dump()}
