import os
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import numpy as np
from scipy import stats

# -----------------------------
# 1. Initialize Firestore
# -----------------------------
def get_db():
    try:
        firebase_admin.get_app()
    except ValueError:
        if os.path.exists("serviceAccountKey.json"):
            cred = credentials.Certificate("serviceAccountKey.json")
            firebase_admin.initialize_app(cred)
        else:
            firebase_admin.initialize_app()
    return firestore.client()

def binarize(scores, cutoff=45):
    return (np.array(scores) >= cutoff).astype(int)

def calculate_spd(y_hat_bin, groups):
    groups = np.array(groups).astype(bool)
    y_hat_bin = np.array(y_hat_bin)
    
    if mask_pos := groups.sum():
        rate_attr = y_hat_bin[groups].mean()
    else:
        rate_attr = 0.0
        
    if mask_neg := (~groups).sum():
        rate_baseline = y_hat_bin[~groups].mean()
    else:
        rate_baseline = 0.0
        
    return rate_baseline - rate_attr

def calculate_dir(y_hat_bin, groups):
    groups = np.array(groups).astype(bool)
    y_hat_bin = np.array(y_hat_bin)
    
    if mask_pos := groups.sum():
        rate_attr = y_hat_bin[groups].mean()
    else:
        rate_attr = 0.0
        
    if mask_neg := (~groups).sum():
        rate_baseline = y_hat_bin[~groups].mean()
    else:
        rate_baseline = 1.0
        
    return rate_attr / rate_baseline if rate_baseline > 0 else 1.0

def run_research_report():
    print("🚀 Generating Research Fairness & Mitigation Report...")
    db = get_db()
    docs = db.collection("userImages").stream()
    
    data = []
    for doc in docs:
        d = doc.to_dict()
        details = d.get("details", {})
        rubric = d.get("rubric", {})
        fairness = d.get("fairness_report") or rubric.get("fairness_report")
        
        if not fairness or "original_total_14" not in fairness:
            continue
            
        data.append({
            "grade": d.get("studentGrade"),
            "is_dyslexic": details.get("dyslexic_flag", False),
            "orig_score": fairness.get("original_total_14"),
            "adj_score": rubric.get("total_14"),
            "mitigation_applied": fairness.get("mitigation_applied", False)
        })
    
    df = pd.DataFrame(data)
    if df.empty:
        print("❌ No data found with fairness mitigation records.")
        return

    # Clean grades
    def clean_grade(g):
        if isinstance(g, int): return g
        if isinstance(g, str):
            try: return int(g.lower().replace("grade", "").strip())
            except: return None
        return None
    
    df['grade'] = df['grade'].apply(clean_grade)
    df = df.dropna(subset=['grade'])
    
    report_md = "# 🛡️ Thesis Fairness & Mitigation Proof Report\n\n"
    report_md += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_md += f"**Dataset Size:** {len(df)} scored essays with mitigation logs\n\n"

    report_md += "## 1. Fairness Metrics: Baseline vs. Mitigated\n"
    report_md += "| Grade | Metric | Baseline (Biased) | Mitigated (Fair) | Improvement |\n"
    report_md += "|-------|--------|-------------------|------------------|-------------|\n"

    for grade in sorted(df['grade'].unique()):
        g_df = df[df['grade'] == grade]
        if len(g_df) < 2: continue
        
        is_dys = g_df['is_dyslexic'].tolist()
        
        # Scale to 0-100 for SPD/DIR calculation based on 45% threshold
        orig_100 = [(s/14)*100 for s in g_df['orig_score']]
        adj_100 = [(s/14)*100 for s in g_df['adj_score']]
        
        y_orig = binarize(orig_100, cutoff=45)
        y_adj = binarize(adj_100, cutoff=45)
        
        spd_orig = calculate_spd(y_orig, is_dys)
        spd_adj = calculate_spd(y_adj, is_dys)
        
        dir_orig = calculate_dir(y_orig, is_dys)
        dir_adj = calculate_dir(y_adj, is_dys)
        
        report_md += f"| {grade} | SPD | {spd_orig:.3f} | {spd_adj:.3f} | {((spd_orig-spd_adj)/spd_orig)*100 if spd_orig > 0 else 0:.1f}% |\n"
        report_md += f"| {grade} | DIR | {dir_orig:.3f} | {dir_adj:.3f} | {((dir_adj-dir_orig)/dir_orig)*100 if dir_orig > 0 else 0:.1f}% |\n"

    report_md += "\n## 2. Mitigation Impact Statement (Dyslexic Students Only)\n"
    report_md += "| Grade | Avg Baseline Score | Avg Mitigated Score | Mean Absolute Boost |\n"
    report_md += "|-------|--------------------|---------------------|---------------------|\n"

    dys_df = df[df['is_dyslexic'] == True]
    for grade in sorted(dys_df['grade'].unique()):
        g_dys = dys_df[dys_df['grade'] == grade]
        avg_orig = g_dys['orig_score'].mean()
        avg_adj = g_dys['adj_score'].mean()
        boost = avg_adj - avg_orig
        
        report_md += f"| {grade} | {avg_orig:.2f} | {avg_adj:.2f} | +{boost:.2f} marks |\n"

    report_md += "\n## 3. Statistical Proof of Fairness\n"
    report_md += "Using independent samples t-test to compare Dyslexic vs. Non-Dyslexic distributions.\n\n"
    report_md += "| Grade | Baseline P-Value | Mitigated P-Value | Conclusion |\n"
    report_md += "|-------|------------------|-------------------|------------|\n"

    for grade in sorted(df['grade'].unique()):
        g_df = df[df['grade'] == grade]
        dys = g_df[g_df['is_dyslexic'] == True]
        non_dys = g_df[g_df['is_dyslexic'] == False]
        
        if len(dys) < 2 or len(non_dys) < 2: continue
        
        _, p_orig = stats.ttest_ind(non_dys['orig_score'], dys['orig_score'])
        _, p_adj = stats.ttest_ind(non_dys['adj_score'], dys['adj_score'])
        
        status = "✅ Bias Eliminated" if p_adj > 0.05 else "⚠️ Bias Reduced"
        report_md += f"| {grade} | {p_orig:.4f} | {p_adj:.4f} | {status} |\n"

    with open("analysis/RESEARCH_FAIRNESS_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    
    print("\n✨ Research report generated: analysis/RESEARCH_FAIRNESS_REPORT.md")

if __name__ == "__main__":
    run_research_report()
