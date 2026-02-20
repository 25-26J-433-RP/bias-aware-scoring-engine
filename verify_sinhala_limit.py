from fastapi.testclient import TestClient
import sys
import os

# Add the project root to sys.path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app.main import app

client = TestClient(app)
VALID_HEADERS = {"X-API-KEY": "akura-research-secret-2026"}

def test_rate_limiting_sinhala_ml():
    """
    Test the actual /score-sinhala-ml endpoint to trigger the 20/minute rate limit.
    """
    print("\n[TEST] Testing rate limiting on /score-sinhala-ml (Limit: 20/min)")
    
    payload = {
        "text": "ශ්‍රී ලංකාවේ පරිසරය රැක ගැනීම අපගේ වගකීමකි.",
        "grade": 7,
        "topic": "පරිසරය",
        "dyslexic_flag": False,
        "error_tags": []
    }
    
    success_count = 0
    limited_count = 0
    limit = 20
    
    # Make 25 requests (limit is 20)
    for i in range(1, limit + 6):
        response = client.post("/score-sinhala-ml", json=payload, headers=VALID_HEADERS)
        
        if response.status_code == 200:
            success_count += 1
            if i % 5 == 0:
                print(f"  - Request {i}: Success (200)")
        elif response.status_code == 429:
            limited_count += 1
            print(f"  - Request {i}: Rate Limited! (429)")
            break
        else:
            print(f"  - Request {i}: Unexpected status {response.status_code}")
            print(response.json())
            break
            
    print(f"\n[RESULTS]")
    print(f"Total Successes: {success_count}")
    print(f"Rate Limited Count: {limited_count}")
    
    if limited_count > 0:
        print("✅ SUCCESS: Rate limiting is working on /score-sinhala-ml!")
    else:
        print("❌ FAILURE: Rate limiting was not triggered.")

if __name__ == "__main__":
    test_rate_limiting_sinhala_ml()
