import sys
import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Add the project root to sys.path so we can import from app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.sinhala_ml_v2 import score_sinhala_ml_v2

def smart_rescore():
    print("=" * 60)
    print("SMART RESCORER V3: ROBUST DATA SEARCH")
    print("=" * 60)
    
    # Initialize Firestore
    try:
        cred = credentials.Certificate("serviceAccountKey.json")
        firebase_admin.initialize_app(cred)
    except:
        pass
        
    db = firestore.client()
    
    print("\nFetching essays from 'userImages'...")
    docs = list(db.collection("userImages").stream())
    print(f"Found {len(docs)} documents.")
    
    updated_count = 0
    skipped_count = 0
    
    for doc in docs:
        doc_id = doc.id
        d = doc.to_dict()
        
        # --- ROBUST SEARCH FOR ESSAY TEXT ---
        essay_text = None
        # 1. Check top level
        if d.get("essay_text"):
            essay_text = d.get("essay_text")
        # 2. Check inside details
        elif d.get("details", {}).get("essay_text"):
            essay_text = d.get("details", {}).get("essay_text")
        # 3. Check inside details/error_tags
        elif d.get("details", {}).get("error_tags", {}).get("essay_text"):
            essay_text = d.get("details", {}).get("error_tags", {}).get("essay_text")
            
        # --- ROBUST SEARCH FOR DYSLEXIC FLAG ---
        dyslexic_flag = d.get("dyslexic_flag")
        if dyslexic_flag is None:
            dyslexic_flag = d.get("details", {}).get("dyslexic_flag", False)
            
        # --- ROBUST SEARCH FOR GRADE ---
        raw_grade = d.get("studentGrade")
        if raw_grade is None:
            raw_grade = d.get("details", {}).get("studentGrade")
            
        if not essay_text or raw_grade is None:
            # print(f"   Skipping {doc_id}: Missing text or grade.")
            skipped_count += 1
            continue
            
        # Parse grade
        try:
            if isinstance(raw_grade, int):
                grade_num = raw_grade
            elif isinstance(raw_grade, str):
                grade_num = int(raw_grade.lower().replace("grade", "").strip())
            else:
                skipped_count += 1
                continue
        except:
            skipped_count += 1
            continue
            
        # Rescore
        try:
            new_result = score_sinhala_ml_v2(essay_text, grade_num, dyslexic_flag)
            
            update_data = {
                "rubric": {
                    "organization_6": new_result.get("organization_6"),
                    "richness_5": new_result.get("richness_5"),
                    "technical_3": new_result.get("technical_3"),
                    "total_14": new_result.get("total_14")
                },
                "score": (new_result.get("total_14", 0) / 14.0) * 100,
                "fairness_report": new_result.get("fairness_report"),
                "model_version": "research_baseline_proportional_v3",
                "updated_at": firestore.SERVER_TIMESTAMP,
                "rescored_at": datetime.utcnow().isoformat()
            }
            
            db.collection("userImages").document(doc_id).update(update_data)
            updated_count += 1
            if updated_count % 50 == 0:
                print(f"   Processed {updated_count}...")
        except:
            skipped_count += 1

    print("\n" + "=" * 60)
    print(f"COMPLETED: Updated {updated_count}, Skipped {skipped_count}")
    print("=" * 60)

if __name__ == "__main__":
    smart_rescore()
