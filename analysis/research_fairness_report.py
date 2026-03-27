import os
from datetime import datetime

import firebase_admin
import numpy as np
import pandas as pd
from firebase_admin import credentials, firestore
from scipy import stats


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

    if groups.sum():
        rate_dys = y_hat_bin[groups].mean()
    else:
        rate_dys = 0.0

    if (~groups).sum():
        rate_non = y_hat_bin[~groups].mean()
    else:
        rate_non = 0.0

    # AIF360-aligned direction: unprivileged - privileged.
    return float(rate_dys - rate_non)


def calculate_dir(y_hat_bin, groups):
    groups = np.array(groups).astype(bool)
    y_hat_bin = np.array(y_hat_bin)

    if groups.sum():
        rate_dys = y_hat_bin[groups].mean()
    else:
        rate_dys = 0.0

    if (~groups).sum():
        rate_non = y_hat_bin[~groups].mean()
    else:
        rate_non = 1.0

    return float(rate_dys / rate_non) if rate_non > 0 else 1.0


def spd_improvement_pct(spd_orig, spd_adj):
    # Fairness target for SPD is 0.
    base = abs(spd_orig)
    if base == 0:
        return 0.0
    return ((base - abs(spd_adj)) / base) * 100.0


def dir_improvement_pct(dir_orig, dir_adj):
    # Fairness target for DIR is 1.
    base = abs(1.0 - dir_orig)
    if base == 0:
        return 0.0
    return ((base - abs(1.0 - dir_adj)) / base) * 100.0


def run_research_report():
    print("Generating Research Fairness & Mitigation Report...")
    db = get_db()
    docs = db.collection("userImages").stream()

    data = []
    for doc in docs:
        d = doc.to_dict()
        details = d.get("details", {})
        rubric = d.get("rubric", {})
        fairness = d.get("fairness_report") or rubric.get("fairness_report")

        # Include BOTH groups:
        # - Dyslexic with mitigation logs: orig from fairness, adj from rubric
        # - Non-dyslexic/no-mitigation: orig == adj == rubric total_14
        adj_score = rubric.get("total_14")
        if adj_score is None:
            continue

        if fairness and "original_total_14" in fairness:
            orig_score = fairness.get("original_total_14")
            mitigation_applied = bool(fairness.get("mitigation_applied", False))
        else:
            orig_score = adj_score
            mitigation_applied = False

        data.append(
            {
                "grade": d.get("studentGrade"),
                "is_dyslexic": details.get("dyslexic_flag", False),
                "orig_score": float(orig_score),
                "adj_score": float(adj_score),
                "mitigation_applied": mitigation_applied,
            }
        )

    df = pd.DataFrame(data)
    if df.empty:
        print("No data found with fairness mitigation records.")
        return

    def clean_grade(g):
        if isinstance(g, int):
            return g
        if isinstance(g, str):
            try:
                return int(g.lower().replace("grade", "").strip())
            except ValueError:
                return None
        return None

    df["grade"] = df["grade"].apply(clean_grade)
    df = df.dropna(subset=["grade"])

    report_md = "# Thesis Fairness & Mitigation Proof Report\n\n"
    report_md += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report_md += f"**Dataset Size:** {len(df)} scored essays with mitigation logs\n\n"

    report_md += "## 1. Fairness Metrics: Baseline vs. Mitigated\n"
    report_md += "| Grade | Metric | Baseline (Biased) | Mitigated (Fair) | Improvement |\n"
    report_md += "|-------|--------|-------------------|------------------|-------------|\n"

    for grade in sorted(df["grade"].unique()):
        g_df = df[df["grade"] == grade]
        if len(g_df) < 2:
            continue

        is_dys = g_df["is_dyslexic"].tolist()

        # Scale total_14 to 0-100, then threshold at 45.
        orig_100 = [(s / 14) * 100 for s in g_df["orig_score"]]
        adj_100 = [(s / 14) * 100 for s in g_df["adj_score"]]

        y_orig = binarize(orig_100, cutoff=45)
        y_adj = binarize(adj_100, cutoff=45)

        spd_orig = calculate_spd(y_orig, is_dys)
        spd_adj = calculate_spd(y_adj, is_dys)
        dir_orig = calculate_dir(y_orig, is_dys)
        dir_adj = calculate_dir(y_adj, is_dys)

        spd_improve = spd_improvement_pct(spd_orig, spd_adj)
        dir_improve = dir_improvement_pct(dir_orig, dir_adj)

        report_md += f"| {grade} | SPD | {spd_orig:.3f} | {spd_adj:.3f} | {spd_improve:+.1f}% |\n"
        report_md += f"| {grade} | DIR | {dir_orig:.3f} | {dir_adj:.3f} | {dir_improve:+.1f}% |\n"

    report_md += "\n## 2. Mitigation Impact Statement (Dyslexic Students Only)\n"
    report_md += "| Grade | Avg Baseline Score | Avg Mitigated Score | Mean Absolute Boost |\n"
    report_md += "|-------|--------------------|---------------------|---------------------|\n"

    dys_df = df[df["is_dyslexic"] == True]
    for grade in sorted(dys_df["grade"].unique()):
        g_dys = dys_df[dys_df["grade"] == grade]
        avg_orig = g_dys["orig_score"].mean()
        avg_adj = g_dys["adj_score"].mean()
        boost = avg_adj - avg_orig
        report_md += f"| {grade} | {avg_orig:.2f} | {avg_adj:.2f} | +{boost:.2f} marks |\n"

    report_md += "\n## 3. Statistical Proof of Fairness\n"
    report_md += "Using independent samples t-test to compare Dyslexic vs. Non-Dyslexic distributions.\n\n"
    report_md += "| Grade | Baseline P-Value | Mitigated P-Value | Conclusion |\n"
    report_md += "|-------|------------------|-------------------|------------|\n"

    for grade in sorted(df["grade"].unique()):
        g_df = df[df["grade"] == grade]
        dys = g_df[g_df["is_dyslexic"] == True]
        non_dys = g_df[g_df["is_dyslexic"] == False]

        if len(dys) < 2 or len(non_dys) < 2:
            report_md += f"| {grade} | N/A | N/A | Insufficient sample for t-test |\n"
            continue

        _, p_orig = stats.ttest_ind(non_dys["orig_score"], dys["orig_score"])
        _, p_adj = stats.ttest_ind(non_dys["adj_score"], dys["adj_score"])

        if np.isnan(p_orig) or np.isnan(p_adj):
            report_md += f"| {grade} | N/A | N/A | Undefined (zero variance / degenerate groups) |\n"
            continue

        status = "Bias Eliminated" if p_adj > 0.05 else "Bias Reduced"
        report_md += f"| {grade} | {p_orig:.4f} | {p_adj:.4f} | {status} |\n"

    with open("analysis/RESEARCH_FAIRNESS_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("Report generated: analysis/RESEARCH_FAIRNESS_REPORT.md")


if __name__ == "__main__":
    run_research_report()
