import pandas as pd
import requests
import json
import firebase_admin
from firebase_admin import credentials, firestore
from tqdm import tqdm
import time

# 1. Setup Firestore (Uses your existing key)
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# 2. Configuration
CSV_PATH = "../test_essays_main.csv"
API_URL = "http://localhost:8000/score-sinhala-ml"
API_KEY = "akura-research-secret-2026"
COLLECTION_NAME = "userImages" # The dashboard reads from here

def run_batch_validation():
    print(f"🚀 Loading test dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # We will mark these as "RESEARCH_BATCH" so you can identify them in Firestore
    batch_tag = f"batch_test_{int(time.time())}"
    
    results = []
    
    print(f"✨ Scoring {len(df)} essays via local AI engine...")
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        payload = {
            "text": row['essay_text'],
            "grade": int(row['grade']),
            "topic": row['essay_topic'],
            "dyslexic_flag": bool(row['dyslexic_flag']),
            "error_tags": []
        }
        
        headers = {
            "X-API-KEY": API_KEY,
            "Content-Type": "application/json"
        }
        
        try:
            # Hit local API
            response = requests.post(API_URL, json=payload, headers=headers)
            if response.status_code == 200:
                api_data = response.json()
                
                # Prepare Firestore Document (matches frontend format)
                firestore_doc = {
                    "image_id": row['essay_id'],
                    "studentGrade": int(row['grade']),
                    "studentGender": "Not Specified",
                    "score": api_data['score'],
                    "details": api_data['details'],
                    "rubric": api_data['rubric'],
                    "fairness_report": api_data.get('fairness_report'),
                    "timestamp": firestore.SERVER_TIMESTAMP,
                    "is_research_test": True,
                    "batch_id": batch_tag
                }
                
                # Save to Firestore
                db.collection(COLLECTION_NAME).document(row['essay_id']).set(firestore_doc)
                results.append(api_data['score'])
            else:
                print(f"❌ Failed for {row['essay_id']}: {response.text}")
        except Exception as e:
            print(f"⚠ Technical Error on {row['essay_id']}: {e}")

    print("\n✅ Batch Processing Complete!")
    print(f"📊 Processed {len(results)} essays.")
    print(f"🔗 All results are now in Firestore under '{COLLECTION_NAME}'.")
    print("\n👉 NEXT STEP: Go to your Dashboard and click 'Run Analysis' to see the new metrics.")

if __name__ == "__main__":
    run_batch_validation()
