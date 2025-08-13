from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_score_contract():
    r = client.post("/score", json={"text": "Sri Lanka environment...", "prompt": "Discuss environment."})
    js = r.json()
    assert r.status_code == 200
    assert "score" in js and 0 <= js["score"] <= 100
    assert "details" in js and "strategy" in js["details"]
    assert set(js["fairness_report"].keys()) == {"spd", "dir", "eod", "mitigation_used"}
