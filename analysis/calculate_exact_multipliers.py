"""
Calculate EXACT calibration multipliers from Firestore data

This script fetches all essays and computes the precise mean scores
for each component (richness, organization, technical) by grade and group.

Output: Exact multipliers to use in sinhala_ml_v2.py
"""

import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firestore
try:
    db = firestore.client()
except ValueError:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()


def fetch_component_scores(grade_filter: int):
    """Fetch component-level scores for a specific grade"""
    docs = db.collection("userImages").stream()
    rows = []

    for doc in docs:
        d = doc.to_dict()

        if "score" not in d or "rubric" not in d:
            continue

        # Parse grade
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

        rubric = d.get("rubric", {})
        details = d.get("details", {})

        rows.append({
            "dyslexic_flag": details.get("dyslexic_flag", False),
            "richness_5": rubric.get("richness_5", 0),
            "organization_6": rubric.get("organization_6", 0),
            "technical_3": rubric.get("technical_3", 0),
            "total_14": rubric.get("total_14", 0),
        })

    return pd.DataFrame(rows)


def calculate_multipliers_for_grade(grade):
    """Calculate exact multipliers for a single grade"""
    df = fetch_component_scores(grade)
    
    if df.empty:
        return None
    
    df_dys = df[df["dyslexic_flag"] == True]
    df_non_dys = df[df["dyslexic_flag"] == False]
    
    if len(df_dys) < 5 or len(df_non_dys) < 5:
        return None
    
    results = {
        "grade": grade,
        "n_dyslexic": len(df_dys),
        "n_non_dyslexic": len(df_non_dys),
        "richness": {
            "mean_dys": df_dys["richness_5"].mean(),
            "mean_non_dys": df_non_dys["richness_5"].mean(),
            "multiplier": df_non_dys["richness_5"].mean() / df_dys["richness_5"].mean() if df_dys["richness_5"].mean() > 0 else 1.0,
        },
        "organization": {
            "mean_dys": df_dys["organization_6"].mean(),
            "mean_non_dys": df_non_dys["organization_6"].mean(),
            "multiplier": df_non_dys["organization_6"].mean() / df_dys["organization_6"].mean() if df_dys["organization_6"].mean() > 0 else 1.0,
        },
        "technical": {
            "mean_dys": df_dys["technical_3"].mean(),
            "mean_non_dys": df_non_dys["technical_3"].mean(),
            "multiplier": df_non_dys["technical_3"].mean() / df_dys["technical_3"].mean() if df_dys["technical_3"].mean() > 0 else 1.0,
        },
        "total": {
            "mean_dys": df_dys["total_14"].mean(),
            "mean_non_dys": df_non_dys["total_14"].mean(),
            "difference": df_non_dys["total_14"].mean() - df_dys["total_14"].mean(),
        }
    }
    
    return results


def main():
    print("\n" + "="*70)
    print("EXACT CALIBRATION MULTIPLIER CALCULATION")
    print("="*70)
    
    all_multipliers = {}
    
    for grade in range(3, 9):
        print(f"\n{'='*70}")
        print(f"GRADE {grade}")
        print(f"{'='*70}")
        
        data = calculate_multipliers_for_grade(grade)
        
        if data is None:
            print(f"[WARNING] Insufficient data for Grade {grade}")
            continue
        
        all_multipliers[grade] = data
        
        print(f"\nSample sizes: Dys={data['n_dyslexic']}, Non-Dys={data['n_non_dyslexic']}")
        
        print("\n--- RICHNESS_5 ---")
        print(f"  Mean Dyslexic: {data['richness']['mean_dys']:.3f}")
        print(f"  Mean Non-Dyslexic: {data['richness']['mean_non_dys']:.3f}")
        print(f"  MULTIPLIER: {data['richness']['multiplier']:.4f}")
        
        print("\n--- ORGANIZATION_6 ---")
        print(f"  Mean Dyslexic: {data['organization']['mean_dys']:.3f}")
        print(f"  Mean Non-Dyslexic: {data['organization']['mean_non_dys']:.3f}")
        print(f"  MULTIPLIER: {data['organization']['multiplier']:.4f}")
        
        print("\n--- TECHNICAL_3 ---")
        print(f"  Mean Dyslexic: {data['technical']['mean_dys']:.3f}")
        print(f"  Mean Non-Dyslexic: {data['technical']['mean_non_dys']:.3f}")
        print(f"  MULTIPLIER: {data['technical']['multiplier']:.4f}")
        
        print("\n--- TOTAL_14 (for reference) ---")
        print(f"  Mean Dyslexic: {data['total']['mean_dys']:.3f}")
        print(f"  Mean Non-Dyslexic: {data['total']['mean_non_dys']:.3f}")
        print(f"  Difference: {data['total']['difference']:.3f}")
    
    # Generate Python code for sinhala_ml_v2.py
    print("\n" + "="*70)
    print("COPY THIS INTO sinhala_ml_v2.py")
    print("="*70)
    print("\nCALIBRATION_MULTIPLIERS = {")
    for grade in range(3, 9):
        if grade in all_multipliers:
            d = all_multipliers[grade]
            print(f'    {grade}: {{')
            print(f'        "richness": {d["richness"]["multiplier"]:.4f},')
            print(f'        "organization": {d["organization"]["multiplier"]:.4f},')
            print(f'        "technical": {d["technical"]["multiplier"]:.4f},')
            print(f'    }},  # n_dys={d["n_dyslexic"]}, n_non_dys={d["n_non_dyslexic"]}')
        else:
            print(f'    {grade}: {{"richness": 1.0, "organization": 1.0, "technical": 1.0}},  # Insufficient data')
    print("}")
    
    print("\n" + "="*70)
    print("[SUCCESS] Exact multipliers calculated")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
