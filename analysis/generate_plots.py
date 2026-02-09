
import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import time
import os

# --- CONFIGURATION ---
CSV_PATH = "c:/Users/nuwan/ResearchProject/test_essays_main.csv"
API_URL = "http://127.0.0.1:8001/score-sinhala-ml"
OUTPUT_IMAGE = "c:/Users/nuwan/ResearchProject/bias-aware-scoring-engine/docs/accuracy_scatterplot.png"

def generate_validation_plots():
    print(f"Loading test data from: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    
    results = []
    
    print(f"Processing {len(df)} essays for plot generation...")
    
    for index, row in df.iterrows():
        payload = {
            "text": row['essay_text'],
            "grade": int(row['grade']),
            "dyslexic_flag": bool(row['dyslexic_flag'])
        }
        
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                data = response.json()
                model_score_14 = data['rubric']['total_14']
                actual_score_14 = row['total_14']
                
                results.append({
                    "Actual Score": actual_score_14,
                    "Model Score": model_score_14,
                    "Grade": row['grade']
                })
        except Exception as e:
            print(f"Failed at {row['essay_id']}: {str(e)}")

    if not results:
        return

    res_df = pd.DataFrame(results)
    
    # Calculate Correlation
    r, _ = pearsonr(res_df['Actual Score'], res_df['Model Score'])
    mae = mean_absolute_error(res_df['Actual Score'], res_df['Model Score'])

    # --- PLOTTING ---
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    # Scatter plot with regression line
    plot = sns.regplot(data=res_df, x="Actual Score", y="Model Score", 
                       scatter_kws={'alpha':0.6, 's':100}, 
                       line_kws={'color':'red', 'lw':2})
    
    plt.title(f"Accuracy Validation: Expert vs. Model Scores\n(Pearson r = {r:.3f}, MAE = {mae:.2f})", 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Expert Assigned Score (Total 14)", fontsize=13)
    plt.ylabel("Model Predicted Score (Total 14)", fontsize=13)
    
    # Add a diagonal line for "Perfect Correlation" reference
    lims = [
        np.min([plt.xlim(), plt.ylim()]),
        np.max([plt.xlim(), plt.ylim()]),
    ]
    plt.plot(lims, lims, 'k--', alpha=0.5, zorder=0, label="Perfect correlation")
    plt.legend()

    # Annotation for research context
    plt.text(0.05, 0.95, f"Empirical Results:\nPearson r: {r:.3f}\nMAE: {mae:.2f}", 
             transform=plot.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300)
    print(f"Plot saved successfully to: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    generate_validation_plots()
