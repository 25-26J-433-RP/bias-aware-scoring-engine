import pytest
from fastapi.testclient import TestClient
from app.main import app

# Initialize TestClient
client = TestClient(app)
VALID_HEADERS = {"X-API-KEY": "akura-research-secret-2026"}

# Sample Sinhala Essay Text (Generic for testing)
TEST_TEXT = "මගේ තාත්තා ඉතා හොඳ පුද්ගලයෙකි. ඔහු අපට ආදරය කරයි. ඔහු වෙහෙස මහන්සි වී වැඩ කරයි."

@pytest.mark.parametrize("grade,dyslexic_flag", [
    (4, False), # Baseline
    (4, True),  # Mitigation
    (8, True),  # High Grade Mitigation
    (6, True)   # Significant but small gap
])
def test_mitigation_logic(grade, dyslexic_flag):
    payload = {
        "text": TEST_TEXT,
        "grade": grade,
        "topic": "My Father",
        "dyslexic_flag": dyslexic_flag
    }
    
    response = client.post("/score-sinhala-ml", json=payload, headers=VALID_HEADERS)
    
    assert response.status_code == 200
    data = response.json()
    rubric = data.get("rubric", {})
    fairness = data.get("fairness_report", {})
    
    # Core assertions
    assert "score" in data
    assert "total_14" in rubric
    
    # Bias Mitigation Logic Check
    if dyslexic_flag:
        # For Grades 4, 6, 8, mitigation should be applied based on empirical analysis
        assert fairness.get("mitigation_applied") is True
        assert fairness.get("total_boost", 0) > 0
        assert "justification" in fairness
    else:
        # Baseline non-dyslexic should not have mitigation
        assert fairness.get("mitigation_applied") is False
        assert fairness.get("total_boost", 0) == 0

if __name__ == "__main__":
    # Allow manual running
    print("Run with: pytest tests/test_mitigation_api.py")
