from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_score_sinhala_ml_contract():
    payload = {
        "text": "ශ්‍රී ලංකාවේ පරිසරය රැක ගැනීම අපගේ වගකීමකි.",
        "grade": 7,
        "topic": "පරිසරය",
        "dyslexic_flag": False,
        "error_tags": []
    }

    r = client.post("/score-sinhala-ml", json=payload)
    assert r.status_code == 200

    js = r.json()

    # Core contract
    assert "score" in js
    assert 0 <= js["score"] <= 100

    # Rubric-level scores
    assert "rubric" in js
    assert "total_14" in js["rubric"]

    # Metadata
    assert "details" in js
    assert js["details"]["grade"] == 7
    assert js["details"]["model"].startswith("xlm-roberta")

    # Fairness report exists but may be None (system-level)
    assert "fairness_report" in js
