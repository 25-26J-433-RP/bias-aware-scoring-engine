"""
Populate Firebase with test data from the training dataset.
This script will:
1. Load essays from the training dataset
2. Score them using the retrained model
3. Upload to Firebase for bias detection testing
"""

import pandas as pd
import torch
from transformers import AutoTokenizer
import firebase_admin
from firebase_admin import credentials, firestore
import sys
from pathlib import Path

# Add parent directory to path to import model
sys.path.append(str(Path(__file__).parent.parent / "scoring-model-training"))
from training.model_multitask_xlmr import SinhalaMultiHeadRegressor

# Configuration
DATASET_PATH = "../scoring-model-training/sinhala_dataset_final_with_dyslexic.csv"
MODEL_PATH = "./models/xlm-roberta-large-sinhala-multihead"
FIREBASE_CRED_PATH = "./serviceAccountKey.json"  # Update this path if needed

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {DEVICE}")


def load_model_and_tokenizer(model_path):
    """Load the retrained model and tokenizer"""
    print(f"📦 Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = SinhalaMultiHeadRegressor.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    
    print("✅ Model loaded successfully")
    return model, tokenizer


def score_essay(model, tokenizer, essay_text, grade):
    """Score a single essay using the model"""
    # Tokenize
    inputs = tokenizer(
        essay_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    grade_id = torch.tensor([grade], dtype=torch.long).to(DEVICE)
    
    # Score
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            grade_id=grade_id
        )
    
    # Extract scores
    scores = {
        "richness_5": float(outputs["richness_5"].cpu().item()),
        "organization_6": float(outputs["organization_6"].cpu().item()),
        "technical_3": float(outputs["technical_3"].cpu().item()),
        "total_14": float(outputs["total_14"].cpu().item())
    }
    
    return scores


def initialize_firebase():
    """Initialize Firebase Admin SDK"""
    print("🔥 Initializing Firebase...")
    
    try:
        # Check if already initialized
        firebase_admin.get_app()
        print("✅ Firebase already initialized")
    except ValueError:
        # Initialize for the first time
        cred = credentials.Certificate(FIREBASE_CRED_PATH)
        firebase_admin.initialize_app(cred)
        print("✅ Firebase initialized")
    
    db = firestore.client()
    return db


def upload_to_firebase(db, essay_data):
    """Upload essay and scores to Firebase"""
    # Create document in userImages collection (matching the analysis script)
    doc_ref = db.collection('userImages').document()
    doc_ref.set(essay_data)
    return doc_ref.id


def main():
    print("=" * 60)
    print("🚀 Populating Firebase with Test Data")
    print("=" * 60)
    
    # Load dataset
    print(f"\n📂 Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    
    # Drop rows with missing values
    df = df.dropna(subset=["grade", "essay_text", "dyslexic_flag"])
    df["grade"] = df["grade"].astype(int)
    
    print(f"✅ Loaded {len(df)} essays")
    print(f"   Dyslexic: {df['dyslexic_flag'].sum()}")
    print(f"   Non-dyslexic: {(~df['dyslexic_flag']).sum()}")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    # Initialize Firebase
    db = initialize_firebase()
    
    # Process essays
    print(f"\n📝 Scoring and uploading essays to Firebase...")
    
    uploaded_count = 0
    for idx, row in df.iterrows():
        try:
            # Score the essay
            scores = score_essay(
                model, 
                tokenizer, 
                row["essay_text"], 
                row["grade"]
            )
            
            # Prepare document matching Firebase structure
            essay_data = {
                # Essay content
                "essay_text": row["essay_text"],
                "essay_topic": row.get("essay_topic", "Test Essay"),
                
                # IMPORTANT: Bias analysis looks for dyslexic_flag INSIDE details
                "details": {
                    "dyslexic_flag": bool(row["dyslexic_flag"]),
                },
                
                # Student metadata (mock data for testing)
                "studentId": f"TEST_STUDENT_{idx}",
                "studentGrade": f"Grade {int(row['grade'])}",
                "studentAge": int(row["grade"]) + 5,  # Approximate age
                "studentGender": "Not specified",
                "userId": "test_user_automated_script",
                
                # Rubric scores from retrained model
                "rubric": {
                    "richness_5": round(scores["richness_5"], 2),
                    "organization_6": round(scores["organization_6"], 2),
                    "technical_3": round(scores["technical_3"], 2),
                    "total_14": round(scores["total_14"], 2)
                },
                
                # IMPORTANT: Normalize score to 0-100 scale for bias detection
                # The bias detection script expects scores > 75 to pass
                "score": round((scores["total_14"] / 14.0) * 100, 2),
                
                # Timestamps
                "uploadedAt": firestore.SERVER_TIMESTAMP,
                "scored_at": firestore.SERVER_TIMESTAMP,
                "updatedAt": firestore.SERVER_TIMESTAMP,
                
                # Image-related fields (null for automated data)
                "filename": None,
                "fileSize": None,
                "imageUrl": None,
                "mimeType": None,
                "storagePath": None,
                
                # Other fields
                "description": "Automated test data from training dataset",
                "error_tags": [],
                "fairness_report": None,
                
                # Metadata
                "source": "automated_script",
                "model_version": "retrained_with_dyslexic_data"
            }
            
            # Upload to Firebase
            doc_id = upload_to_firebase(db, essay_data)
            uploaded_count += 1
            
            if uploaded_count % 50 == 0:
                print(f"   Uploaded {uploaded_count}/{len(df)} essays...")
        
        except Exception as e:
            print(f"⚠️ Error processing essay {idx}: {e}")
            continue
    
    print(f"\n✅ Upload complete!")
    print(f"   Total uploaded: {uploaded_count} essays")
    print(f"   Dyslexic: {df['dyslexic_flag'].sum()}")
    print(f"   Non-dyslexic: {(~df['dyslexic_flag']).sum()}")
    
    print("\n" + "=" * 60)
    print("🎉 Firebase populated successfully!")
    print("=" * 60)
    print("\n📊 Next step: Run bias detection")
    print("   Command: python analysis\\firestore_fairness_eval.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
