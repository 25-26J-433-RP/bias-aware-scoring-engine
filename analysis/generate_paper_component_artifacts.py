import re
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(r"c:\Users\nuwan\ResearchProject\bias-aware-scoring-engine")
CSV_PATH = Path(r"c:\Users\nuwan\ResearchProject\Akura dataset human vs AI score - Sheet1.csv")
FAIRNESS_REPORT = ROOT / "analysis" / "RESEARCH_FAIRNESS_REPORT.md"
FIG_DIR = ROOT / "analysis" / "figures"
TABLE_DIR = ROOT / "analysis" / "paper_tables"


def find_header_row(raw_df: pd.DataFrame) -> int:
    for i, row in raw_df.iterrows():
        line = " | ".join(map(str, row.tolist()))
        has_grade = ("Grade" in line) or ("grade" in line)
        has_ai = ("AI" in line) or ("ai" in line)
        has_teacher = ("Teacher" in line) or ("teacher" in line)
        if has_grade and has_ai and has_teacher:
            return i
    raise ValueError("Header row not found in teacher vs AI CSV.")


def find_column(columns: list[str], must_have: list[str]) -> str:
    for col in columns:
        if all(token in col for token in must_have):
            return col
    raise ValueError(f"Could not find column with tokens: {must_have}")


def compute_scoring_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(CSV_PATH, header=None)
    header_row = find_header_row(raw)
    df = pd.read_csv(CSV_PATH, header=header_row)
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

    ai_col = find_column(df.columns.tolist(), ["AI", "Total/14"])
    teacher_col = find_column(df.columns.tolist(), ["Teacher", "Total/14"])
    grade_col = find_column(df.columns.tolist(), ["Grade"])

    df[grade_col] = pd.to_numeric(df[grade_col], errors="coerce")
    df[ai_col] = pd.to_numeric(df[ai_col], errors="coerce")
    df[teacher_col] = pd.to_numeric(df[teacher_col], errors="coerce")

    filtered = df[
        (df[grade_col] >= 3)
        & (df[grade_col] <= 8)
        & df[ai_col].notna()
        & df[teacher_col].notna()
    ].copy()

    ai = filtered[ai_col].to_numpy(float)
    teacher = filtered[teacher_col].to_numpy(float)

    overall = pd.DataFrame(
        [
            {
                "Metric": "Rows used",
                "Value": len(filtered),
            },
            {
                "Metric": "Pearson r",
                "Value": round(float(np.corrcoef(ai, teacher)[0, 1]), 3),
            },
            {
                "Metric": "Spearman r",
                "Value": round(float(spearmanr(ai, teacher).correlation), 3),
            },
            {
                "Metric": "MAE (/14)",
                "Value": round(float(np.mean(np.abs(ai - teacher))), 3),
            },
            {
                "Metric": "RMSE (/14)",
                "Value": round(float(sqrt(np.mean((ai - teacher) ** 2))), 3),
            },
        ]
    )

    by_grade_rows = []
    for grade in sorted(filtered[grade_col].dropna().unique()):
        gdf = filtered[filtered[grade_col] == grade]
        if len(gdf) < 2:
            continue
        g_ai = gdf[ai_col].to_numpy(float)
        g_teacher = gdf[teacher_col].to_numpy(float)
        by_grade_rows.append(
            {
                "Grade": int(grade),
                "Count": len(gdf),
                "Pearson r": round(float(np.corrcoef(g_ai, g_teacher)[0, 1]), 3),
                "MAE (/14)": round(float(np.mean(np.abs(g_ai - g_teacher))), 3),
            }
        )

    return overall, pd.DataFrame(by_grade_rows)


def parse_fairness_report() -> tuple[pd.DataFrame, pd.DataFrame]:
    text = FAIRNESS_REPORT.read_text(encoding="utf-8")

    fairness_rows = []
    impact_rows = []

    fairness_pattern = re.compile(
        r"\|\s*(\d+)\s*\|\s*(SPD|DIR)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*([+\-\d.]+)%\s*\|"
    )
    impact_pattern = re.compile(
        r"\|\s*(\d+)\s*\|\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|\s*\+?([-\d.]+)\s*marks\s*\|"
    )

    for grade, metric, before, after, improvement in fairness_pattern.findall(text):
        fairness_rows.append(
            {
                "Grade": int(grade),
                "Metric": metric,
                "Before": float(before),
                "After": float(after),
                "Improvement (%)": float(improvement),
            }
        )

    for grade, before, after, boost in impact_pattern.findall(text):
        impact_rows.append(
            {
                "Grade": int(grade),
                "Avg Baseline Score": float(before),
                "Avg Mitigated Score": float(after),
                "Mean Absolute Boost": float(boost),
            }
        )

    fairness_df = pd.DataFrame(fairness_rows)
    impact_df = pd.DataFrame(impact_rows)
    return fairness_df, impact_df


def plot_fairness_chart(fairness_df: pd.DataFrame) -> Path:
    spd_df = fairness_df[fairness_df["Metric"] == "SPD"].sort_values("Grade")
    dir_df = fairness_df[fairness_df["Metric"] == "DIR"].sort_values("Grade")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    w = 0.36

    x_spd = np.arange(len(spd_df))
    axes[0].bar(x_spd - w / 2, spd_df["Before"], width=w, label="Before", color="#c44e52")
    axes[0].bar(x_spd + w / 2, spd_df["After"], width=w, label="After", color="#55a868")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticks(x_spd)
    axes[0].set_xticklabels(spd_df["Grade"])
    axes[0].set_title("SPD by Grade")
    axes[0].set_xlabel("Grade")
    axes[0].set_ylabel("SPD")
    axes[0].legend()

    x_dir = np.arange(len(dir_df))
    axes[1].bar(x_dir - w / 2, dir_df["Before"], width=w, label="Before", color="#8172b2")
    axes[1].bar(x_dir + w / 2, dir_df["After"], width=w, label="After", color="#64b5cd")
    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].axhline(0.8, color="red", linewidth=1, linestyle="--", label="0.8 threshold")
    axes[1].set_xticks(x_dir)
    axes[1].set_xticklabels(dir_df["Grade"])
    axes[1].set_title("DIR by Grade")
    axes[1].set_xlabel("Grade")
    axes[1].set_ylabel("DIR")
    axes[1].legend()

    plt.tight_layout()
    out_path = FIG_DIR / "paper_fairness_before_after.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def write_markdown_table(df: pd.DataFrame, path: Path, title: str) -> None:
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(df.columns.astype(str)) + " |")
    lines.append("|" + "|".join(["---"] * len(df.columns)) + "|")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(map(str, row.tolist())) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    overall_df, grade_df = compute_scoring_metrics()
    fairness_df, impact_df = parse_fairness_report()
    fairness_chart = plot_fairness_chart(fairness_df)

    overall_df.to_csv(TABLE_DIR / "paper_scoring_overall_metrics.csv", index=False)
    grade_df.to_csv(TABLE_DIR / "paper_scoring_by_grade.csv", index=False)
    fairness_df.to_csv(TABLE_DIR / "paper_fairness_before_after.csv", index=False)
    impact_df.to_csv(TABLE_DIR / "paper_dyslexic_boost_by_grade.csv", index=False)

    write_markdown_table(
        overall_df, TABLE_DIR / "paper_scoring_overall_metrics.md", "Paper Scoring Overall Metrics"
    )
    write_markdown_table(
        grade_df, TABLE_DIR / "paper_scoring_by_grade.md", "Paper Scoring Metrics by Grade"
    )
    write_markdown_table(
        fairness_df, TABLE_DIR / "paper_fairness_before_after.md", "Paper Fairness Before/After"
    )
    write_markdown_table(
        impact_df, TABLE_DIR / "paper_dyslexic_boost_by_grade.md", "Paper Dyslexic Boost by Grade"
    )

    print("Generated paper artifacts:")
    print(f"- Existing scoring figure: {FIG_DIR / 'ai_vs_teacher_comparison.png'}")
    print(f"- New fairness figure: {fairness_chart}")
    print(f"- Tables written to: {TABLE_DIR}")


if __name__ == "__main__":
    main()
