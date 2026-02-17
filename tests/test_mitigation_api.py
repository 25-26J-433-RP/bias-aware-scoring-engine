import requests
import json

# Configuration
BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/score-sinhala-ml"

# Sample Sinhala Essay Text
TEST_TEXT = "මගේ තාත්තා ඉතා හොඳ පුද්ගලයෙකි. ඔහු අපට ආදරය කරයි. ඔහු වෙහෙස මහන්සි වී වැඩ කරයි."

def test_scoring(grade, dyslexic_flag):
    payload = {
        "text": TEST_TEXT,
        "grade": grade,
        "topic": "My Father",
        "dyslexic_flag": dyslexic_flag
    }
    
    print(f"\n[TEST] Testing Grade {grade} | Dyslexic: {dyslexic_flag}")
    print("-" * 50)
    
    headers = {
        "X-API-KEY": "akura-research-secret-2026"
    }
    
    try:
        response = requests.post(ENDPOINT, json=payload, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            rubric = data.get("rubric", {})
            fairness = data.get("fairness_report", {})
            
            print(f"SUCCESS")
            print(f"Total Score (14): {rubric.get('total_14')}")
            print(f"Breakdown: R={rubric.get('richness_5')} | O={rubric.get('organization_6')} | T={rubric.get('technical_3')}")
            
            if fairness.get("mitigation_applied"):
                print(f"MITIGATION APPLIED: YES")
                print(f"   Boost Amount: +{fairness.get('total_boost', 0)} points")
                print(f"   Justification: {fairness.get('justification')}")
            else:
                print(f"MITIGATION APPLIED: NO")
                
            return data
        else:
            print(f"FAILED (Status {response.status_code})")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return None

if __name__ == "__main__":
    print("STARTING RESEARCH VALIDATION TEST")
    
    # 1. Test Grade 4 Non-Dyslexic (Baseline)
    non_dys = test_scoring(grade=4, dyslexic_flag=False)
    
    # 2. Test Grade 4 Dyslexic (Should trigger mitigation)
    dys = test_scoring(grade=4, dyslexic_flag=True)
    
    # 3. Test Grade 8 (Should show smaller mitigation)
    test_scoring(grade=8, dyslexic_flag=True)
    
    # 4. Test Grade 6 (Statistically significant but small gap)
    test_scoring(grade=6, dyslexic_flag=True)

    print("\n" + "="*50)
    print("TEST COMPLETE")
    print("="*50)
