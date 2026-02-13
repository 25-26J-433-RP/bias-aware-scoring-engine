from pydantic import BaseModel, Field
from typing import Optional, List


# -----------------------------
# English Essay Scoring
# -----------------------------
class EssayIn(BaseModel):
    text: str
    prompt: Optional[str] = None


class FairnessReport(BaseModel):
    spd: float
    dir: float
    eod: Optional[float] = None
    mitigation_used: Optional[str] = None


class ScoreOut(BaseModel):
    score: float
    details: dict
    fairness_report: Optional[FairnessReport] = None


# -----------------------------
# Sinhala Scoring Inputs
# -----------------------------
class SinhalaEssayIn(BaseModel):
    text: str = Field(..., min_length=10, max_length=5000, description="Sinhala essay text")
    grade: Optional[int] = Field(None, ge=3, le=8, description="Grade level (3-8)")
    topic: Optional[str] = None
    dyslexic_flag: bool = False
    error_tags: Optional[List[str]] = None


# -----------------------------
# Multi-head Rubric Output
# -----------------------------
class SinhalaRubricOut(BaseModel):
    richness_5: Optional[float] = None
    organization_6: Optional[float] = None
    technical_3: Optional[float] = None
    total_14: Optional[float] = None


class SinhalaMLOut(BaseModel):
    score: float
    details: dict
    rubric: SinhalaRubricOut
    fairness_report: Optional[FairnessReport] = None


# -----------------------------
# Batch Fairness Evaluation
# -----------------------------
class FairnessEvalIn(BaseModel):
    score: float = Field(..., ge=0, le=100)
    y_true: int = Field(..., description="1 = pass, 0 = fail")
    dyslexic_flag: bool
    grade: int = Field(..., ge=3, le=8)
