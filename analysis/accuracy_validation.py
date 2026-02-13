
import pandas as pd
import requests
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import time

# --- CONFIGURATION ---
CSV_PATH = "c:/Users/nuwan/ResearchProject/test_essays_main.csv"
# Using the Local Backend for reliability
API_URL = "http://127.0.0.1:8001/score-sinhala-ml"

def validate_accuracy():
    print(f"Loading test data from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    results = []
    
    print(f"Processing {len(df)} essays... This may take a moment.")
    
    for index, row in df.iterrows():
        payload = {
            "text": row['essay_text'],
            "grade": int(row['grade']),
            "dyslexic_flag": bool(row['dyslexic_flag'])
        }
        
        try:
            start_time = time.time()
            response = requests.post(API_URL, json=payload)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                # We want the total_14 score from the rubric
                model_score_14 = data['rubric']['total_14']
                actual_score_14 = row['total_14']
                
                results.append({
                    "id": row['essay_id'],
                    "actual": actual_score_14,
                    "model": model_score_14,
                    "latency": latency
                })
                print(f"Done [{index+1}/{len(df)}] {row['essay_id']}: Actual={actual_score_14}, Model={model_score_14:.2f} ({latency:.2f}s)")
            else:
                print(f"Error at {row['essay_id']}: {response.text}")
        except Exception as e:
            print(f"Failed to connect to API: {str(e)}")
            break

    if not results:
        return

    # Convert to DataFrame for calculations
    res_df = pd.DataFrame(results)
    
    # --- METRICS CALCULATION ---
    r, p_value = pearsonr(res_df['actual'], res_df['model'])
    mae = mean_absolute_error(res_df['actual'], res_df['model'])
    avg_latency = res_df['latency'].mean()
    
    print("\n" + "="*40)
    print("ACCURACY VALIDATION SUMMARY")
    print("="*40)
    print(f"Total Essays Processed: {len(res_df)}")
    print(f"Pearson Correlation (r): {r:.4f}")
    print(f"P-Value: {p_value:.4e}")
    print(f"Mean Absolute Error (MAE): {mae:.4f} marks")
    print(f"Average Latency: {avg_latency:.2f} seconds")
    print("="*40)
    
    if r >= 0.90:
        print("SUCCESS: Target correlation (>= 0.90) achieved!")
    else:
        print("WARNING: Correlation is below target. Model calibration needed.")

if __name__ == "__main__":
    validate_accuracy()
