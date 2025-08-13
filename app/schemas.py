from pydantic import BaseModel, Field

class EssayIn(BaseModel):
    text: str = Field(..., min_length=1, description="Normalized Sinhala/English essay text")
    prompt: str | None = Field(None, description="Optional essay prompt/question")

class FairnessReport(BaseModel):
    spd: float
    dir: float
    eod: float
    mitigation_used: str | None = None

class ScoreOut(BaseModel):
    score: float
    details: dict
    fairness_report: FairnessReport
