import argparse
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_CSV = r"c:\Users\nuwan\ResearchProject\Akura dataset human vs AI score - Sheet1.csv"


def find_header_row(raw_df: pd.DataFrame) -> int:
    for i, row in raw_df.iterrows():
        line = " | ".join(map(str, row.tolist()))
        has_grade = ("Grade" in line) or ("grade" in line)
        has_ai = ("AI" in line) or ("ai" in line)
        has_teacher = ("Teacher" in line) or ("teacher" in line)
        if has_grade and has_ai and has_teacher:
            return i
    raise ValueError("Header row not found. Export sheet again as CSV and retry.")


def find_column(columns: list[str], must_have: list[str]) -> str:
    for col in columns:
        if all(token in col for token in must_have):
            return col
    raise ValueError(f"Could not find column with tokens: {must_have}. Found columns: {columns}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute AI vs Teacher scoring metrics.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to CSV file")
    parser.add_argument("--grade-min", type=int, default=6, help="Minimum grade to include")
    parser.add_argument("--grade-max", type=int, default=8, help="Maximum grade to include")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    raw = pd.read_csv(csv_path, header=None)
    header_row = find_header_row(raw)

    df = pd.read_csv(csv_path, header=header_row)
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

    ai_col = find_column(df.columns.tolist(), ["AI", "Total/14"])
    teacher_col = find_column(df.columns.tolist(), ["Teacher", "Total/14"])
    grade_col = find_column(df.columns.tolist(), ["Grade"])

    df[grade_col] = pd.to_numeric(df[grade_col], errors="coerce")
    df[ai_col] = pd.to_numeric(df[ai_col], errors="coerce")
    df[teacher_col] = pd.to_numeric(df[teacher_col], errors="coerce")

    filtered = df[
        (df[grade_col] >= args.grade_min)
        & (df[grade_col] <= args.grade_max)
        & df[ai_col].notna()
        & df[teacher_col].notna()
    ].copy()

    if filtered.empty:
        raise ValueError("No valid rows after filtering. Check grades or CSV content.")

    ai = filtered[ai_col].to_numpy(float)
    teacher = filtered[teacher_col].to_numpy(float)

    pearson = float(np.corrcoef(ai, teacher)[0, 1])
    mae = float(np.mean(np.abs(ai - teacher)))
    rmse = float(sqrt(np.mean((ai - teacher) ** 2)))
    spearman = spearmanr(ai, teacher)

    ai_round = np.rint(ai)
    pearson_round = float(np.corrcoef(ai_round, teacher)[0, 1])
    mae_round = float(np.mean(np.abs(ai_round - teacher)))

    print("=" * 64)
    print("AI vs Teacher Metrics")
    print("=" * 64)
    print(f"CSV: {csv_path}")
    print(f"Rows used: {len(filtered)}")
    print(f"Grade range: {args.grade_min}-{args.grade_max}")
    print(f"Columns: AI='{ai_col}' | Teacher='{teacher_col}' | Grade='{grade_col}'")
    print("-" * 64)
    print(f"Pearson:  {pearson:.3f}")
    print(f"MAE:      {mae:.3f} / 14")
    print(f"RMSE:     {rmse:.3f} / 14")
    print(f"Spearman: {spearman.correlation:.3f}  (p={spearman.pvalue:.6f})")
    print("-" * 64)
    print(f"Pearson (rounded AI): {pearson_round:.3f}")
    print(f"MAE (rounded AI):     {mae_round:.3f} / 14")
    print("=" * 64)


if __name__ == "__main__":
    main()
