from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from .schemas import EssayIn, ScoreOut
from .scorer import score_essay
from .fairness import demo_fairness_report
from .sinhala_baseline import baseline_sinhala_score
from .schemas import SinhalaEssayIn

app = FastAPI(title="Bias-Aware Scoring Engine", version="0.1.0")

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

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")

@app.post("/score", response_model=ScoreOut)
def score(payload: EssayIn):
    score_value, details = score_essay(payload.text, payload.prompt)
    fr = demo_fairness_report()  # Phase 2: synthetic fairness numbers
    return {"score": round(score_value, 2), "details": details, "fairness_report": fr.model_dump()}

@app.get("/fairness/smoke", tags=["fairness"])
def fairness_smoke():
    return demo_fairness_report()

@app.post("/score-sinhala", response_model=ScoreOut)
def score_sinhala(payload: SinhalaEssayIn):
    score_value, details = baseline_sinhala_score(payload.text)
    return {
        "score": score_value,
        "details": details,
        "fairness_report": None  # now allowed because ScoreOut makes it optional
    }

