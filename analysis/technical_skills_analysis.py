
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- CONFIGURATION ---
CSV_PATH = "c:/Users/nuwan/ResearchProject/test_essays_main.csv"
API_URL = "http://127.0.0.1:8001/score-sinhala-ml"
OUTPUT_IMAGE = "c:/Users/nuwan/ResearchProject/bias-aware-scoring-engine/docs/technical_skills_bias_analysis.png"

def analyze_technical_bias():
    print("Starting Technical Skills Discrepancy Analysis...")
    df = pd.read_csv(CSV_PATH)
    
    results = []
    
    for index, row in df.iterrows():
        payload = {
            "text": row['essay_text'],
            "grade": int(row['grade']),
            "dyslexic_flag": False # We set to False to see the RAW model bias before mitigation
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                data = response.json()
                model_tech = data['rubric']['technical_3']
                expert_tech = row['technical_3']
                
                results.append({
                    "id": row['essay_id'],
                    "is_dyslexic": row['dyslexic_flag'],
                    "grade": row['grade'],
                    "expert_tech": expert_tech,
                    "model_tech": model_tech,
                    "error": model_tech - expert_tech
                })
                print(f"Processed {row['essay_id']}")
        except Exception as e:
            print(f"Error at {row['essay_id']}: {e}")

    res_df = pd.DataFrame(results)
    
    # Calculate Gaps
    dyslexic_gap = res_df[res_df['is_dyslexic'] == True]['error'].mean()
    non_dyslexic_gap = res_df[res_df['is_dyslexic'] == False]['error'].mean()
    
    print("\n" + "="*40)
    print("TECHNICAL SKILLS BIAS SUMMARY")
    print("="*40)
    print(f"Avg Error (Non-Dyslexic): {non_dyslexic_gap:.3f}")
    print(f"Avg Error (Dyslexic): {dyslexic_gap:.3f}")
    print(f"Disparity (Bias Gap): {abs(dyslexic_gap - non_dyslexic_gap):.3f}")
    print("="*40)

    # --- VISUALIZATION ---
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Boxplot of errors by Dyslexia Status
    plt.subplot(1, 2, 1)
    sns.boxplot(data=res_df, x="is_dyslexic", y="error", palette="Set2")
    plt.title("Technical Score Error Distribution\n(Model - Expert)", fontsize=12, fontweight='bold')
    plt.ylabel("Score Difference (Marks)")
    plt.xlabel("Dyslexic Student Flag")

    # Bar chart of Mean technical scores
    plt.subplot(1, 2, 2)
    means = res_df.groupby('is_dyslexic')[['expert_tech', 'model_tech']].mean().reset_index()
    means_melted = means.melt(id_vars="is_dyslexic", var_name="Scorer", value_name="Score")
    sns.barplot(data=means_melted, x="is_dyslexic", y="Score", hue="Scorer", palette="viridis")
    plt.title("Mean Technical Scores Comparison", fontsize=12, fontweight='bold')
    plt.ylabel("Technical Score (Out of 3)")
    plt.ylim(0, 3)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Analysis plot saved to: {OUTPUT_IMAGE}")

    # Save summary text for paper
    with open("c:/Users/nuwan/ResearchProject/bias-aware-scoring-engine/docs/technical_bias_summary.txt", "w") as f:
        f.write(f"TECHNICAL SKILLS BIAS ANALYSIS REPORT\n")
        f.write(f"Sample Size: {len(res_df)}\n")
        f.write(f"Non-Dyslexic Avg Technical Error: {non_dyslexic_gap:.3f}\n")
        f.write(f"Dyslexic Avg Technical Error: {dyslexic_gap:.3f}\n")
        f.write(f"Evidence of Penalty: The model penalizes dyslexic students by an additional {abs(dyslexic_gap - non_dyslexic_gap):.3f} marks on average compared to non-dyslexic peers.\n")

if __name__ == "__main__":
    analyze_technical_bias()
