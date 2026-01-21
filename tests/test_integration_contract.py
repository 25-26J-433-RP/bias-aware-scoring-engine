import sys
import os
from fastapi.testclient import TestClient

# Add project root to path so we can import app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

def test_score_essay_contract_dyslexic_flag_true():
    """
    Test the contract for scoring an essay with the dyslexic flag set to True.
    This simulates the full incoming JSON structure expected from the Classifier/Reconstruction components.
    """
    print("\n--- Testing Dyslexic Flag Contract (True) ---")
    payload = {
        "text": "මෙම රචනාව පරීක්ෂා කිරීම සඳහා වේ.", # Sinhala dummy text
        "grade": 5,
        "topic": "පාසල් නිවාඩුව",
        "dyslexic_flag": True,
        "error_tags": ["mirror_writing", "phonetic_error"]
    }
    
    response = client.post("/score-sinhala-ml", json=payload)
    
    # Assertions for Contract
    if response.status_code != 200:
        print(f"FAILED: Status {response.status_code}")
        print(response.text)
        raise AssertionError("API call failed")

    data = response.json()
    
    # Check output structure matches expectation (Contract)
    assert "score" in data, "Missing 'score' field"
    assert "rubric" in data, "Missing 'rubric' field"
    assert "details" in data, "Missing 'details' field"
    
    # Check if input flag was respected and echoed back
    # This proves the backend received and parsed the flag correctly
    assert data["details"]["dyslexic_flag"] is True, "dyslexic_flag was not preserved in details"
    
    print("✅ Contract verified: Payload with dyslexic_flag=True accepted and processed.")
    print("Response snippet:", data["details"])

def test_score_essay_contract_dyslexic_flag_false():
    """
    Test the contract for scoring an essay with the dyslexic flag set to False.
    """
    print("\n--- Testing Dyslexic Flag Contract (False) ---")
    payload = {
        "text": "සාමාන්‍ය රචනාවක්.",
        "grade": 6,
        "dyslexic_flag": False
    }
    response = client.post("/score-sinhala-ml", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["details"]["dyslexic_flag"] is False
    print("✅ Contract verified: Payload with dyslexic_flag=False accepted.")

if __name__ == "__main__":
    # Allow running as a script
    try:
        test_score_essay_contract_dyslexic_flag_true()
        test_score_essay_contract_dyslexic_flag_false()
        print("\n🎉 ALL CONTRACT TESTS PASSED")
    except AssertionError as e:
        print(f"\n❌ Contract broken: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        exit(1)
