import pandas as pd
import requests
import json

API_URL = "http://127.0.0.1:8000/score-sinhala"

def clean_text(t):
    """Fix multiline Sinhala input issues automatically."""
    if isinstance(t, str):
        return t.replace("\n", "\\n").replace("\r", "\\r").strip()
    return ""

def score_row(row):
    payload = {
        "text": clean_text(row["essay_text"]),
        "grade": int(row["grade"]),
        "topic": row["essay_topic"]
    }

    try:
        response = requests.post(API_URL, json=payload)
        data = response.json()
        return data.get("score", None)
    except Exception as e:
        print("Error scoring", row["essay_id"], e)
        return None

def main():
    print("Loading dataset...")
    df = pd.read_csv("Book(Sheet1).csv")  # your uploaded CSV name

    print("Scoring essays...")
    df["baseline_score"] = df.apply(score_row, axis=1)

    print("Saving output...")
    df.to_csv("scored_output.csv", index=False)

    print("DONE! Output saved as scored_output.csv")

if __name__ == "__main__":
    main()
