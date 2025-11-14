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
    fairness_report: FairnessReport | None = None

class SinhalaEssayIn(BaseModel):
    text: str = Field(..., min_length=1, description="Raw Sinhala essay text")
    grade: int = Field(..., ge=3, le=8, description="Grade level 3–8")
    topic: str | None = Field(None, description="Essay topic/title")

