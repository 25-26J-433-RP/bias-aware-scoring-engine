import argparse
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEFAULT_CSV = r"c:\Users\nuwan\ResearchProject\Akura dataset human vs AI score - Sheet1.csv"
DEFAULT_OUT = r"c:\Users\nuwan\ResearchProject\bias-aware-scoring-engine\analysis\figures\ai_vs_teacher_comparison.png"


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


def load_filtered_data(csv_path: Path, grade_min: int, grade_max: int) -> tuple[pd.DataFrame, str, str, str]:
    raw = pd.read_csv(csv_path, header=None)
    header_row = find_header_row(raw)
    df = pd.read_csv(csv_path, header=header_row)
    df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]

    ai_col = find_column(df.columns.tolist(), ["AI", "Total/14"])
    teacher_col = find_column(df.columns.tolist(), ["Teacher", "Total/14"])
    grade_col = find_column(df.columns.tolist(), ["Grade"])

    essay_col = next((c for c in df.columns if "Essay" in c and "Topic" not in c), None)
    if essay_col is None:
        essay_col = "Essay_ID"
        df[essay_col] = [f"E{i + 1:02d}" for i in range(len(df))]

    df[grade_col] = pd.to_numeric(df[grade_col], errors="coerce")
    df[ai_col] = pd.to_numeric(df[ai_col], errors="coerce")
    df[teacher_col] = pd.to_numeric(df[teacher_col], errors="coerce")

    filtered = df[
        (df[grade_col] >= grade_min)
        & (df[grade_col] <= grade_max)
        & df[ai_col].notna()
        & df[teacher_col].notna()
    ].copy()

    if filtered.empty:
        raise ValueError("No valid rows after filtering. Check grades or CSV content.")

    filtered = filtered.sort_values([grade_col, essay_col]).reset_index(drop=True)
    return filtered, ai_col, teacher_col, essay_col


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI vs Teacher comparison chart.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to CSV file")
    parser.add_argument("--grade-min", type=int, default=3, help="Minimum grade to include")
    parser.add_argument("--grade-max", type=int, default=8, help="Maximum grade to include")
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output PNG path")
    parser.add_argument(
        "--mode",
        choices=["comparison", "error_focus"],
        default="comparison",
        help="comparison = score bars; error_focus = absolute-gap view",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    filtered, ai_col, teacher_col, essay_col = load_filtered_data(
        csv_path, args.grade_min, args.grade_max
    )

    ai = filtered[ai_col].to_numpy(float)
    teacher = filtered[teacher_col].to_numpy(float)

    pearson = float(np.corrcoef(ai, teacher)[0, 1])
    mae = float(np.mean(np.abs(ai - teacher)))
    rmse = float(sqrt(np.mean((ai - teacher) ** 2)))
    spearman = spearmanr(ai, teacher)

    dys_col = next((c for c in filtered.columns if "Dyslexic" in c), None)
    if dys_col:
        group_series = filtered[dys_col].astype(str).str.lower()
        is_dys = group_series.str.contains("dyslexic") & ~group_series.str.contains("non")
        is_non = group_series.str.contains("non")
    else:
        is_dys = pd.Series([False] * len(filtered))
        is_non = pd.Series([True] * len(filtered))

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1.4], hspace=0.35, wspace=0.25)
    ax_scatter = fig.add_subplot(gs[:, 0])
    ax_dys = fig.add_subplot(gs[0, 1])
    ax_non = fig.add_subplot(gs[1, 1])

    # Left: scatter with separate groups
    if is_non.any():
        ax_scatter.scatter(
            teacher[is_non.values],
            ai[is_non.values],
            c="#1b9e77",
            s=75,
            alpha=0.9,
            edgecolors="black",
            linewidth=0.4,
            label="Non-dyslexic",
        )
    if is_dys.any():
        ax_scatter.scatter(
            teacher[is_dys.values],
            ai[is_dys.values],
            c="#d95f02",
            s=75,
            alpha=0.9,
            edgecolors="black",
            linewidth=0.4,
            label="Dyslexic",
        )
    ax_scatter.plot([0, 14], [0, 14], linestyle="--", color="gray", linewidth=1.2, label="Ideal (y=x)")
    ax_scatter.set_xlim(0, 14)
    ax_scatter.set_ylim(0, 14)
    ax_scatter.set_xlabel("Teacher Score (/14)")
    ax_scatter.set_ylabel("AI Score (/14)")
    ax_scatter.set_title("Overall Alignment (Colored by Group)")
    ax_scatter.grid(alpha=0.2)
    ax_scatter.legend(loc="lower right", fontsize=9)

    # Metrics text (overall headline + subgroup MAE only)
    lines = [
        f"Overall N={len(filtered)} | r={pearson:.3f} | MAE={mae:.3f}",
    ]
    if is_dys.any():
        dys_ai = ai[is_dys.values]
        dys_t = teacher[is_dys.values]
        dys_mae = np.mean(np.abs(dys_ai - dys_t))
        lines.append(f"Dyslexic N={len(dys_ai)} | MAE={dys_mae:.3f}")
    if is_non.any():
        non_ai = ai[is_non.values]
        non_t = teacher[is_non.values]
        non_mae = np.mean(np.abs(non_ai - non_t))
        lines.append(f"Non-dys N={len(non_ai)} | MAE={non_mae:.3f}")
    lines.append(f"RMSE={rmse:.3f} | Spearman={spearman.correlation:.3f}")
    ax_scatter.text(
        0.03,
        0.97,
        "\n".join(lines),
        transform=ax_scatter.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.92},
    )

    def draw_group_bar(ax, mask, title):
        gdf = filtered[mask].copy()
        if gdf.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_title(title)
            ax.set_axis_off()
            return
        x_labels = gdf[essay_col].astype(str).tolist()
        t_vals = gdf[teacher_col].to_numpy(float)
        a_vals = gdf[ai_col].to_numpy(float)
        mae_group = float(np.mean(np.abs(a_vals - t_vals)))
        x = np.arange(len(x_labels))
        width = 0.42
        ax.bar(x - width / 2, t_vals, width=width, label="Teacher", color="#4c78a8")
        ax.bar(x + width / 2, a_vals, width=width, label="AI", color="#f58518")
        ax.set_ylim(0, 14)
        ax.set_ylabel("Score (/14)")
        ax.set_title(f"{title} (MAE={mae_group:.2f} / 14)")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=9)

    def draw_group_error(ax, mask, title):
        gdf = filtered[mask].copy()
        if gdf.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_title(title)
            ax.set_axis_off()
            return
        x_labels = gdf[essay_col].astype(str).tolist()
        gaps = np.abs(gdf[ai_col].to_numpy(float) - gdf[teacher_col].to_numpy(float))
        mae_group = float(np.mean(gaps))
        x = np.arange(len(x_labels))
        ax.bar(x, gaps, color="#7b6cff")
        ax.axhline(mae_group, color="red", linestyle="--", linewidth=1.3, label=f"MAE={mae_group:.2f}")
        ax.set_ylim(0, max(3.0, float(np.max(gaps)) + 0.5))
        ax.set_ylabel("Absolute Gap |AI - Teacher|")
        ax.set_title(f"{title} (Lower is better)")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=9)

    if args.mode == "comparison":
        draw_group_bar(ax_dys, is_dys, "Dyslexic Essays: AI vs Teacher")
        draw_group_bar(ax_non, is_non, "Non-dyslexic Essays: AI vs Teacher")
    else:
        draw_group_error(ax_dys, is_dys, "Dyslexic Essays: Absolute Error")
        draw_group_error(ax_non, is_non, "Non-dyslexic Essays: Absolute Error")

    title_suffix = "Score Comparison" if args.mode == "comparison" else "Error-Focused View"
    fig.suptitle(
        f"Grade {args.grade_min}-{args.grade_max}: AI vs Teacher Validation ({title_suffix})",
        fontsize=15,
        y=0.98,
    )
    fig.subplots_adjust(top=0.91)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Chart saved: {out_path}")


if __name__ == "__main__":
    main()
