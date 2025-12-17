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
        "evaluated_at": firestore.SERVER_TIMESTAMP,
        "data_source": "Firestore:userImages",
        "notes": "Only successfully scored essays included"
    })


# -----------------------------
# 4. Run grade-wise fairness evaluation
# -----------------------------
def run_fairness_eval():
    print("\n📊 GRADE-WISE FAIRNESS EVALUATION (Grades 3–8)")
    print("------------------------------------------------")

    for grade in range(3, 9):  # Grades 3 to 8
        df = fetch_user_images(grade_filter=grade)

        if df.empty:
            print(f"\n⚠️ No data available for Grade {grade}")
            continue

        scores = df["score"].tolist()
        groups = df["dyslexic_flag"].tolist()

        y_hat = binarize(scores, cutoff=75)

        report = {
            "grade": grade,
            "spd": round(spd(y_hat, groups), 3),
            "dir": round(dir_ratio(y_hat, groups), 3),
            "threshold": 75,
            "sample_size": len(df),
        }

        print(f"\n📌 Grade {grade}")
        for k, v in report.items():
            print(f"{k}: {v}")

        store_fairness_report(report)

    print("\n✅ Fairness evaluation completed for all grades.")


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    run_fairness_eval()
