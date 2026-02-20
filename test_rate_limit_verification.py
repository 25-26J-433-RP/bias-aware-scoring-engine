from fastapi.testclient import TestClient
import time
import sys
import os

# Add the project root to sys.path to allow imports from app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from app.main import app

client = TestClient(app)

def test_rate_limiting_score_endpoint():
    """
    Simultaneously test the /score endpoint to trigger the 30/minute rate limit.
    """
    print("\n[TEST] Testing rate limiting on /score (Limit: 30/min)")
    
    # We'll use a valid but simple payload for /score
    payload = {
        "text": "The environment is important for our future.",
        "prompt": "Write about the environment."
    }
    
    success_count = 0
    limited_count = 0
    
    # Make 35 requests (limit is 30)
    for i in range(1, 41):
        response = client.post("/score", json=payload)
        
        if response.status_code == 200:
            success_count += 1
            if i % 10 == 0:
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
        print("SUCCESS: Rate limiting is working!")
    else:
        print("FAILURE: Rate limiting was not triggered.")

if __name__ == "__main__":
    test_rate_limiting_score_endpoint()
