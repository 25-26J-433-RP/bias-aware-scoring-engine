from pydantic import BaseModel, Field
from typing import Optional, List


# -----------------------------
# English Essay Scoring 
# -----------------------------
class EssayIn(BaseModel):
    text: str = Field(..., min_length=1, description="Normalized Sinhala/English essay text")
    prompt: Optional[str] = Field(None, description="Optional essay prompt/question")


class FairnessReport(BaseModel):
    spd: float
    dir: float
    eod: float
    mitigation_used: Optional[str] = None


class ScoreOut(BaseModel):
    score: float
    details: dict
    fairness_report: Optional[FairnessReport] = None


# -----------------------------
# Sinhala Baseline & ML Scoring
# -----------------------------
class SinhalaEssayIn(BaseModel):
    """
    Input schema for Sinhala scoring pipeline.
    Accepts:
        - reconstructed Sinhala essay text
        - grade (3–8)
        - topic
        - dyslexia flag (from Tiny's classifier)
        - error tags (from Hasindu/Tiny upstream modules)
    """
    text: str = Field(..., min_length=1, description="Reconstructed Sinhala essay text")
    grade: int = Field(..., ge=3, le=8, description="Student grade (3–8)")
    topic: Optional[str] = Field(None, description="Essay topic or title")
    dyslexic_flag: bool = Field(False, description="Whether the student shows dyslexic writing patterns")
    error_tags: Optional[List[str]] = Field(
        None,
        description="List of dyslexia-related error tags detected by the pattern classifier"
    )


class SinhalaMLIn(BaseModel):
    text: str = Field(..., min_length=1, description="Sinhala essay text (cleaned/reconstructed)")
    topic: Optional[str] = Field(None, description="Essay topic for ML scoring")


# ============================================================
# 🔵 PHASE 5 — Rubric Multi-Head Output Structure
# ============================================================
class SinhalaRubricOut(BaseModel):
    """
    Multi-head rubric score structure:
        - richness (5 marks)
        - organization (6 marks)
        - technical accuracy (3 marks)
        - total (14 marks)
    Currently None (placeholders) until ML model is trained.
    """
    richness_5: Optional[float] = None
    organization_6: Optional[float] = None
    technical_3: Optional[float] = None
    total_14: Optional[float] = None


class SinhalaMLOut(BaseModel):
    """
    Output schema for Sinhala ML scoring.
    Returns:
        - total ML score (0–100)
        - details block
        - rubric multi-head block
        - fairness report (SPD, DIR, EOD)
    """
    score: float
    details: dict
    rubric: SinhalaRubricOut
    fairness_report: Optional[FairnessReport] = None
