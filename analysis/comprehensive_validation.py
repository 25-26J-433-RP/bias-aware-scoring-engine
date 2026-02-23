import requests
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
import time

CSV_PATH = "c:/Users/nuwan/ResearchProject/test_essays_main.csv"
API_URL = "http://127.0.0.1:8001/score-sinhala-ml"
API_KEY = "akura-research-secret-2026"

def run_comprehensive_validation():
    print(f"🚀 Starting Comprehensive Accuracy Validation...")
    print(f"📂 Loading data from {CSV_PATH}")
    
    df = pd.read_csv(CSV_PATH)
    results = []
    
    for index, row in df.iterrows():
        payload = {
            "text": row['essay_text'],
            "grade": int(row['grade']),
            "topic": row['essay_topic'],
            "dyslexic_flag": bool(row['dyslexic_flag'])
        }
        
        try:
            start_time = time.time()
            headers = {"X-API-KEY": API_KEY}
            response = requests.post(API_URL, json=payload, headers=headers)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                res = {
                    "id": row['essay_id'],
                    "grade": row['grade'],
                    "dyslexic": row['dyslexic_flag'],
                    "actual_total": row['total_14'],
                    "model_total": data['rubric']['total_14'],
                    "actual_rich": row['richness_5'],
                    "model_rich": data['rubric']['richness_5'],
                    "actual_org": row['organization_6'],
                    "model_org": data['rubric']['organization_6'],
                    "actual_tech": row['technical_3'],
                    "model_tech": data['rubric']['technical_3'],
                    "latency": latency
                }
                results.append(res)
                print(f"✅ {row['essay_id']} processed.")
            else:
                print(f"❌ Error at {row['essay_id']}: {response.status_code}")
        except Exception as e:
            print(f"🔥 Connection failed: {str(e)}")
            break

    if not results:
        return

    res_df = pd.DataFrame(results)
    
    # Global Metrics
    r, p = pearsonr(res_df['actual_total'], res_df['model_total'])
    mae = mean_absolute_error(res_df['actual_total'], res_df['model_total'])
    
    # Per Grade Metrics
    grade_report = []
    for g in sorted(res_df['grade'].unique()):
        g_df = res_df[res_df['grade'] == g]
        if len(g_df) > 1:
            gr, gp = pearsonr(g_df['actual_total'], g_df['model_total'])
        else:
            gr, gp = 0, 0
        g_mae = mean_absolute_error(g_df['actual_total'], g_df['model_total'])
        grade_report.append({
            "grade": g,
            "count": len(g_df),
            "correlation": gr,
            "mae": g_mae
        })

    # Save Report
    with open("analysis/FINAL_ACCURACY_REPORT.md", "w", encoding="utf-8") as f:
        f.write("# 🔬 Thesis Accuracy Validation Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 📊 Summary Metrics\n")
        f.write(f"- **Total Essays:** {len(res_df)}\n")
        f.write(f"- **Pearson Correlation (r):** `{r:.4f}`\n")
        f.write(f"- **P-Value:** `{p:.4e}`\n")
        f.write(f"- **Mean Absolute Error (MAE):** `{mae:.4f} / 14.0`\n")
        f.write(f"- **Average Latency:** `{res_df['latency'].mean():.2f}s`\n\n")
        
        f.write("## 🏫 Performance by Grade\n")
        f.write("| Grade | Count | Correlation (r) | MAE |\n")
        f.write("|-------|-------|-----------------|-----|\n")
        for gr in grade_report:
            f.write(f"| {gr['grade']} | {gr['count']} | {gr['correlation']:.4f} | {gr['mae']:.4f} |\n")
        
        f.write("\n## 🔍 Component Analysis\n")
        r_rich, _ = pearsonr(res_df['actual_rich'], res_df['model_rich'])
        r_org, _ = pearsonr(res_df['actual_org'], res_df['model_org'])
        r_tech, _ = pearsonr(res_df['actual_tech'], res_df['model_tech'])
        f.write(f"- **Richness Correlation:** `{r_rich:.4f}`\n")
        f.write(f"- **Organization Correlation:** `{r_org:.4f}`\n")
        f.write(f"- **Technical Correlation:** `{r_tech:.4f}`\n\n")
        
        f.write("## ⚠️ Findings\n")
        if r < 0.90:
            f.write("- **[GAP]** Pearson correlation is below the 0.90 target. Current: " + f"{r:.4f}\n")
        else:
            f.write("- **[SUCCESS]** Met target correlation of 0.90.\n")
        
        if mae > 1.5:
            f.write("- **[ISSUE]** MAE is high (>1.5). The model is systematically under/over scoring compared to human experts.\n")

    print("\n✨ Comprehensive report generated: analysis/FINAL_ACCURACY_REPORT.md")

if __name__ == "__main__":
    run_comprehensive_validation()
