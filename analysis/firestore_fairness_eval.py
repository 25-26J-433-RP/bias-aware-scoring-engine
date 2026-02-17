import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from scipy import stats  # For statistical significance testing

from app.fairness import binarize, spd, dir_ratio

# =================================================
# NOTE:
# Equal Opportunity Difference (EOD) requires
# teacher-annotated ground truth labels.
# These are NOT available in Firestore.
# Therefore, EOD is intentionally NOT computed here.
# =================================================


# -----------------------------
# 1. Initialize Firestore
# -----------------------------
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()


# -----------------------------
# 2. Fetch batch essays (by grade)
# -----------------------------
def fetch_user_images(grade_filter: int):
    docs = db.collection("userImages").stream()
    rows = []

    for doc in docs:
        d = doc.to_dict()

        # Required fields
        if "score" not in d or "details" not in d:
            continue

        # -------- grade parsing (ROBUST) --------
        raw_grade = d.get("studentGrade")

        if raw_grade is None:
            continue

        try:
            if isinstance(raw_grade, int):
                grade = raw_grade
            elif isinstance(raw_grade, str):
                grade = int(raw_grade.lower().replace("grade", "").strip())
            else:
                continue
        except ValueError:
            continue

        if grade != grade_filter:
            continue

        details = d.get("details", {})

        rows.append({
            "doc_id": doc.id,
            "score": d["score"],
            "dyslexic_flag": details.get("dyslexic_flag", False),
            "gender": d.get("studentGender"),
            "grade": grade,
        })

    return pd.DataFrame(rows)


# -------------------------------------------------
# 3. Store fairness results as separate documents
# -------------------------------------------------
def store_fairness_report(report: dict):
    doc_id = f"grade_{report['grade']}_{datetime.utcnow().strftime('%Y%m%d')}"

    db.collection("fairnessReports").document(doc_id).set({
        "grade": report["grade"],
        "spd": report["spd"],
        "dir": report["dir"],
        "protected_attribute": "dyslexic_flag",
        "threshold": report["threshold"],
        "sample_size": report["sample_size"],
        # Research-level calibration data
        "n_dyslexic": report.get("n_dyslexic", 0),
        "n_non_dyslexic": report.get("n_non_dyslexic", 0),
        "mean_dyslexic": report.get("mean_dyslexic", 0.0),
        "mean_non_dyslexic": report.get("mean_non_dyslexic", 0.0),
        "calibration_multiplier": report.get("calibration_multiplier", 1.0),
        # New Research Metrics (Added 2026-02-18)
        "t_statistic": report.get("t_statistic", 0.0),
        "p_value": report.get("p_value", 1.0),
        "cohens_d": report.get("cohens_d", 0.0),
        "effect_size": report.get("effect_size", "none"),
        "statistically_significant": report.get("statistically_significant", False),
        # Metadata
        "evaluated_at": firestore.SERVER_TIMESTAMP,
        "data_source": "Firestore:userImages",
        "notes": "Sri Lankan Standard Threshold (45%) applied"
    })


# -----------------------------
# 4. Run grade-wise fairness evaluation
# -----------------------------
def run_fairness_eval():
    print("\n[FAIRNESS] GRADE-WISE FAIRNESS EVALUATION (Grades 3-8)")
    print("------------------------------------------------")

    for grade in range(3, 9):  # Grades 3 to 8
        df = fetch_user_images(grade_filter=grade)

        if df.empty:
            print(f"\n[WARNING] No data available for Grade {grade}")
            continue

        scores = df["score"].tolist()
        groups = df["dyslexic_flag"].tolist()

        # Sri Lankan education standard: 45% is passing grade
        y_hat = binarize(scores, cutoff=45)

        # Calculate group-wise statistics for calibration
        dyslexic_scores = df[df["dyslexic_flag"] == True]["score"].tolist()
        non_dyslexic_scores = df[df["dyslexic_flag"] == False]["score"].tolist()
        
        mean_dyslexic = round(sum(dyslexic_scores) / len(dyslexic_scores), 3) if dyslexic_scores else 0.0
        mean_non_dyslexic = round(sum(non_dyslexic_scores) / len(non_dyslexic_scores), 3) if non_dyslexic_scores else 0.0
        
        # Calculate calibration multiplier (for proportional post-processing)
        # Formula: multiplier = mean_non_dyslexic / mean_dyslexic
        # This ensures dyslexic scores are scaled to match non-dyslexic distribution
        if mean_dyslexic > 0 and mean_non_dyslexic > 0:
            calibration_multiplier = round(mean_non_dyslexic / mean_dyslexic, 4)
        else:
            calibration_multiplier = 1.0

        # Statistical significance testing
        # Perform independent samples t-test
        if len(dyslexic_scores) >= 2 and len(non_dyslexic_scores) >= 2:
            t_statistic, p_value = stats.ttest_ind(non_dyslexic_scores, dyslexic_scores)
            
            # Calculate effect size (Cohen's d)
            # d = (mean1 - mean2) / pooled_std
            pooled_std = (
                ((len(dyslexic_scores) - 1) * stats.tstd(dyslexic_scores)**2 +
                 (len(non_dyslexic_scores) - 1) * stats.tstd(non_dyslexic_scores)**2) /
                (len(dyslexic_scores) + len(non_dyslexic_scores) - 2)
            )**0.5
            
            cohens_d = (mean_non_dyslexic - mean_dyslexic) / pooled_std if pooled_std > 0 else 0.0
            
            # Interpret effect size
            # |d| < 0.2: negligible, 0.2-0.5: small, 0.5-0.8: medium, >0.8: large
            if abs(cohens_d) < 0.2:
                effect_interpretation = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_interpretation = "small"
            elif abs(cohens_d) < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
            
            # Interpret p-value
            is_significant = p_value < 0.05
        else:
            t_statistic = 0.0
            p_value = 1.0
            cohens_d = 0.0
            effect_interpretation = "insufficient_data"
            is_significant = False

        report = {
            "grade": grade,
            "spd": round(spd(y_hat, groups), 3),
            "dir": round(dir_ratio(y_hat, groups), 3),
            "threshold": 45,  # Sri Lankan passing grade
            "sample_size": len(df),
            "n_dyslexic": len(dyslexic_scores),
            "n_non_dyslexic": len(non_dyslexic_scores),
            "mean_dyslexic": mean_dyslexic,
            "mean_non_dyslexic": mean_non_dyslexic,
            "calibration_multiplier": calibration_multiplier,
            # Statistical significance
            "t_statistic": round(t_statistic, 3),
            "p_value": round(p_value, 4),
            "cohens_d": round(cohens_d, 3),
            "effect_size": effect_interpretation,
            "statistically_significant": bool(is_significant),
        }

        print(f"\n[GRADE {grade}]")
        for k, v in report.items():
            print(f"{k}: {v}")

        store_fairness_report(report)

    print("\n[SUCCESS] Fairness evaluation completed for all grades.")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    run_fairness_eval()
