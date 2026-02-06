import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

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
        "mean_dyslexic": report.get("mean_dyslexic", 0.0),
        "mean_non_dyslexic": report.get("mean_non_dyslexic", 0.0),
        "n_dyslexic": report.get("n_dyslexic", 0),
        "n_non_dyslexic": report.get("n_non_dyslexic", 0),
        "calibration_offset": report.get("calibration_offset", 0.0),
        "calibration_multiplier": report.get("calibration_multiplier", 1.0),
        "evaluated_at": firestore.SERVER_TIMESTAMP,
        "data_source": "Firestore:userImages",
        "notes": "Only successfully scored essays included"
    })


# -----------------------------
# 4. Run grade-wise fairness evaluation
# -----------------------------
def run_fairness_eval():
    print("\nGRADE-WISE FAIRNESS EVALUATION (Grades 3-8)")
    print("------------------------------------------------")

    for grade in range(3, 9):  # Grades 3 to 8
        df = fetch_user_images(grade_filter=grade)

        if df.empty:
            print(f"\n No data available for Grade {grade}")
            continue

        scores = df["score"].tolist()
        groups = df["dyslexic_flag"].tolist()

        y_hat = binarize(scores, cutoff=75)
        
        # Calculate mean scores per group (needed for calibration)
        dyslexic_scores = df[df["dyslexic_flag"] == True]["score"]
        non_dyslexic_scores = df[df["dyslexic_flag"] == False]["score"]
        
        mean_dyslexic = float(dyslexic_scores.mean()) if len(dyslexic_scores) > 0 else 0.0
        mean_non_dyslexic = float(non_dyslexic_scores.mean()) if len(non_dyslexic_scores) > 0 else 0.0
        
        # Calculate calibration for mitigation
        calibration_offset = mean_non_dyslexic - mean_dyslexic
        # Multiplier logic: protects against inflating short essays
        mean_d_safe = max(1.0, mean_dyslexic)
        calibration_multiplier = mean_non_dyslexic / mean_d_safe

        report = {
            "grade": grade,
            "spd": round(spd(y_hat, groups), 3),
            "dir": round(dir_ratio(y_hat, groups), 3),
            "threshold": 75,
            "sample_size": len(df),
            # New fields for conditional mitigation
            "mean_dyslexic": round(mean_dyslexic, 2),
            "mean_non_dyslexic": round(mean_non_dyslexic, 2),
            "n_dyslexic": len(dyslexic_scores),
            "n_non_dyslexic": len(non_dyslexic_scores),
            "calibration_offset": round(calibration_offset, 2),
            "calibration_multiplier": round(calibration_multiplier, 3),
        }

        print(f"\nGrade {grade}")
        for k, v in report.items():
            print(f"   {k}: {v}")
        
        # Check if mitigation would be triggered (ONLY for unfavorable bias)
        # Unfavorable = dyslexic students scoring LOWER
        spd_unfavorable = report["spd"] < -0.1  # Negative SPD = dyslexic disadvantaged
        dir_unfavorable = report["dir"] < 0.8   # DIR < 0.8 = dyslexic disadvantaged
        
        if spd_unfavorable or dir_unfavorable:
            print(f"   UNFAVORABLE BIAS DETECTED - Dyslexic students scoring lower")
            print(f"      Mitigation will apply Proportional Boost: x{report['calibration_multiplier']:.3f}")
        elif report["dir"] > 1.25:
            print(f"   Favorable bias detected (dyslexic scoring higher) - NO mitigation applied")
            print(f"      Dyslexic students will be scored like normal students.")
        else:
            print(f"   No significant bias - Dyslexic students scored like normal students")

        store_fairness_report(report)

    print("\nFairness evaluation completed for all grades.")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    run_fairness_eval()
