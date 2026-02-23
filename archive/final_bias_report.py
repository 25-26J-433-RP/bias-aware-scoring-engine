
import pandas as pd
import requests
import numpy as np

# --- CONFIGURATION ---
CSV_PATH = "c:/Users/nuwan/ResearchProject/test_essays_main.csv"
API_URL = "http://127.0.0.1:8001/score-sinhala-ml"

def final_bias_report():
    df = pd.read_csv(CSV_PATH)
    results = []
    
    for _, row in df.iterrows():
        try:
            response = requests.post(API_URL, json={"text": row['essay_text'], "grade": int(row['grade']), "dyslexic_flag": False})
            if response.status_code == 200:
                results.append({
                    "is_dyslexic": row['dyslexic_flag'],
                    "expert_tech": row['technical_3'],
                    "model_tech": response.json()['rubric']['technical_3']
                })
        except: pass

    res_df = pd.DataFrame(results)
    
    # Calculate group means
    stats = res_df.groupby('is_dyslexic').mean()
    
    print("\n" + "="*50)
    print("FINAL BIAS DISCOVERY REPORT")
    print("="*50)
    print(f"EXPERT (Baseline) Technical Scores:")
    print(f"  - Non-Dyslexic: {stats.loc[False, 'expert_tech']:.2f}")
    print(f"  - Dyslexic:     {stats.loc[True, 'expert_tech']:.2f}")
    print(f"  - EXPERT GAP:   {stats.loc[False, 'expert_tech'] - stats.loc[True, 'expert_tech']:.2f}")
    
    print(f"\nMODEL (Raw) Technical Scores:")
    print(f"  - Non-Dyslexic: {stats.loc[False, 'model_tech']:.2f}")
    print(f"  - Dyslexic:     {stats.loc[True, 'model_tech']:.2f}")
    print(f"  - MODEL GAP:    {stats.loc[False, 'model_tech'] - stats.loc[True, 'model_tech']:.2f}")
    
    improvement = (stats.loc[False, 'expert_tech'] - stats.loc[True, 'expert_tech']) - (stats.loc[False, 'model_tech'] - stats.loc[True, 'model_tech'])
    print(f"\nRESEARCH FINDING: The model's inherent bias is {improvement:.2f} marks LOWER than the expert labels.")
    print("="*50)

if __name__ == "__main__":
    final_bias_report()
