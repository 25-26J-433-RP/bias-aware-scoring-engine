import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.mitigation import mitigator

client = TestClient(app)
VALID_HEADERS = {"X-API-KEY": "akura-research-secret-2026"}

TEST_TEXT = (
    "මගේ තාත්තා ඉතා හොඳ පුද්ගලයෙක්. ඔහු අපට ආදරය කරයි. "
    "ඔහු වෙහෙස මහන්සි වී වැඩ කරයි."
)


@pytest.fixture(autouse=True)
def _reset_mitigation_state(monkeypatch):
    monkeypatch.setenv("FAIRNESS_STATIC_FALLBACK", "0")
    old_metrics = dict(mitigator.grade_metrics)
    old_active = dict(mitigator.mitigation_active)
    old_mult = dict(mitigator.calibration_multipliers)
    yield
    mitigator.grade_metrics = old_metrics
    mitigator.mitigation_active = old_active
    mitigator.calibration_multipliers = old_mult


def _score(grade: int, dyslexic_flag: bool):
    payload = {
        "text": TEST_TEXT,
        "grade": grade,
        "topic": "My Father",
        "dyslexic_flag": dyslexic_flag,
    }
    response = client.post("/score-sinhala-ml", json=payload, headers=VALID_HEADERS)
    assert response.status_code == 200
    return response.json()


def test_non_dyslexic_never_adjusted():
    data = _score(grade=4, dyslexic_flag=False)
    fairness = data["rubric"]["fairness_report"]
    assert fairness["mitigation_applied"] is False
    assert fairness.get("total_boost", 0) == 0


def test_dyslexic_adjusted_when_dashboard_thresholds_are_violated():
    mitigator.update_fairness_metrics(
        4,
        {
            "spd": -0.2,
            "dir": 0.62,
            "sample_size": 80,
            "mean_dyslexic": 50.0,
            "mean_non_dyslexic": 58.0,
            "calibration_multiplier": 1.10,
        },
    )

    data = _score(grade=4, dyslexic_flag=True)
    fairness = data["rubric"]["fairness_report"]
    assert fairness["mitigation_applied"] is True
    assert fairness["total_boost"] > 0
    assert fairness["fallback_used"] is False


def test_dyslexic_not_adjusted_when_thresholds_not_violated():
    mitigator.update_fairness_metrics(
        6,
        {
            "spd": -0.02,
            "dir": 0.95,
            "sample_size": 120,
            "mean_dyslexic": 56.0,
            "mean_non_dyslexic": 57.0,
            "calibration_multiplier": 1.01,
        },
    )

    data = _score(grade=6, dyslexic_flag=True)
    fairness = data["rubric"]["fairness_report"]
    assert fairness["mitigation_applied"] is False
    assert fairness["total_boost"] == 0
