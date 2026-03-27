import os
from datetime import datetime

import firebase_admin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from firebase_admin import credentials, firestore


OUT_DIR = os.path.join("analysis", "figures")
PASS_CUTOFF = 45.0


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


def clean_grade(g):
    if isinstance(g, int):
        return g
    if isinstance(g, str):
        try:
            return int(g.lower().replace("grade", "").strip())
        except ValueError:
            return None
    return None


def binarize(scores, cutoff=PASS_CUTOFF):
    s = np.array(scores, dtype=float)
    return (s >= cutoff).astype(int)


def spd(y_bin, group_bool):
    y = np.array(y_bin)
    g = np.array(group_bool, dtype=bool)
    rate_dys = y[g].mean() if g.sum() else 0.0
    rate_non = y[~g].mean() if (~g).sum() else 0.0
    return float(rate_dys - rate_non)


def dir_ratio(y_bin, group_bool):
    y = np.array(y_bin)
    g = np.array(group_bool, dtype=bool)
    rate_dys = y[g].mean() if g.sum() else 0.0
    rate_non = y[~g].mean() if (~g).sum() else 1.0
    return float(rate_dys / rate_non) if rate_non > 0 else 1.0


def fetch_user_images_df(db):
    rows = []
    for doc in db.collection("userImages").stream():
        d = doc.to_dict()
        details = d.get("details", {})
        rubric = d.get("rubric", {}) or {}
        fairness = d.get("fairness_report") or rubric.get("fairness_report") or {}

        grade = clean_grade(d.get("studentGrade"))
        if grade is None:
            continue

        adj_total_14 = rubric.get("total_14")
        if adj_total_14 is None:
            continue

        if "original_total_14" in fairness:
            orig_total_14 = fairness.get("original_total_14")
        else:
            # Non-mitigated/non-dys records: baseline == adjusted.
            orig_total_14 = adj_total_14

        rows.append(
            {
                "doc_id": doc.id,
                "grade": grade,
                "is_dyslexic": bool(details.get("dyslexic_flag", False)),
                "orig_total_14": float(orig_total_14),
                "adj_total_14": float(adj_total_14),
                "orig_100": float(orig_total_14) / 14.0 * 100.0,
                "adj_100": float(adj_total_14) / 14.0 * 100.0,
            }
        )
    return pd.DataFrame(rows)


def fetch_fairness_reports_df(db):
    rows = []
    for doc in db.collection("fairnessReports").stream():
        d = doc.to_dict()
        grade = clean_grade(d.get("grade"))
        if grade is None:
            continue
        ts = d.get("evaluated_at")
        if ts is not None and hasattr(ts, "replace"):
            # Firestore timestamp supports to datetime-like conversion
            try:
                evaluated_at = ts.replace(tzinfo=None)
            except TypeError:
                evaluated_at = datetime.utcnow()
        else:
            evaluated_at = datetime.utcnow()

        rows.append(
            {
                "doc_id": doc.id,
                "grade": grade,
                "evaluated_at": evaluated_at,
                "spd": d.get("spd"),
                "dir": d.get("dir"),
                "sample_size": d.get("sample_size"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["grade", "evaluated_at"])
    return df


def make_spd_dir_before_after(df):
    records = []
    for grade in sorted(df["grade"].unique()):
        gdf = df[df["grade"] == grade]
        groups = gdf["is_dyslexic"].tolist()
        y_orig = binarize(gdf["orig_100"].tolist(), PASS_CUTOFF)
        y_adj = binarize(gdf["adj_100"].tolist(), PASS_CUTOFF)
        records.append(
            {
                "grade": grade,
                "SPD_before": spd(y_orig, groups),
                "SPD_after": spd(y_adj, groups),
                "DIR_before": dir_ratio(y_orig, groups),
                "DIR_after": dir_ratio(y_adj, groups),
            }
        )
    out = pd.DataFrame(records)
    out.to_csv(os.path.join(OUT_DIR, "spd_dir_before_after_by_grade.csv"), index=False)
    return out


def plot_spd_dir_before_after(tbl):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    x = np.arange(len(tbl))
    w = 0.38

    axes[0].bar(x - w / 2, tbl["SPD_before"], w, label="Before")
    axes[0].bar(x + w / 2, tbl["SPD_after"], w, label="After")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("SPD by Grade (Before vs After)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tbl["grade"])
    axes[0].set_xlabel("Grade")
    axes[0].set_ylabel("SPD")
    axes[0].legend()

    axes[1].bar(x - w / 2, tbl["DIR_before"], w, label="Before")
    axes[1].bar(x + w / 2, tbl["DIR_after"], w, label="After")
    axes[1].axhline(1.0, color="black", linewidth=1)
    axes[1].axhline(0.8, color="red", linestyle="--", linewidth=1, label="0.8 threshold")
    axes[1].set_title("DIR by Grade (Before vs After)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tbl["grade"])
    axes[1].set_xlabel("Grade")
    axes[1].set_ylabel("DIR")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "spd_dir_before_after_by_grade.png"), dpi=300)
    plt.close(fig)


def plot_mean_before_after_dyslexic_only(df):
    ddf = df[df["is_dyslexic"] == True].copy()
    if ddf.empty:
        return

    rows = []
    for grade in sorted(ddf["grade"].unique()):
        g = ddf[ddf["grade"] == grade]
        rows.append(
            {
                "grade": grade,
                "before_100": g["orig_100"].mean(),
                "after_100": g["adj_100"].mean(),
            }
        )

    mdf = pd.DataFrame(rows)
    mdf.to_csv(os.path.join(OUT_DIR, "mean_scores_before_after_dyslexic_only.csv"), index=False)

    long_df = mdf.melt(
        id_vars=["grade"],
        value_vars=["before_100", "after_100"],
        var_name="phase",
        value_name="mean_score_100",
    )
    long_df["phase"] = long_df["phase"].map({"before_100": "Before", "after_100": "After"})

    plt.figure(figsize=(10.5, 5.2))
    sns.barplot(data=long_df, x="grade", y="mean_score_100", hue="phase", errorbar=None)
    plt.title("Dyslexic Students: Mean Score by Grade (Before vs After Mitigation)")
    plt.xlabel("Grade")
    plt.ylabel("Mean Score (0-100)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "mean_before_after_dyslexic_only.png"), dpi=300)
    plt.close()


def plot_spd_dir_trend(fairness_df):
    if fairness_df.empty:
        return

    # SPD trend
    plt.figure(figsize=(12, 5.5))
    for grade, gdf in fairness_df.groupby("grade"):
        plt.plot(gdf["evaluated_at"], gdf["spd"], marker="o", label=f"Grade {grade}")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Monthly SPD Trend by Grade")
    plt.xlabel("Evaluation Date")
    plt.ylabel("SPD")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "trend_spd_by_grade.png"), dpi=300)
    plt.close()

    # DIR trend
    plt.figure(figsize=(12, 5.5))
    for grade, gdf in fairness_df.groupby("grade"):
        plt.plot(gdf["evaluated_at"], gdf["dir"], marker="o", label=f"Grade {grade}")
    plt.axhline(1.0, color="black", linewidth=1)
    plt.axhline(0.8, color="red", linestyle="--", linewidth=1)
    plt.title("Monthly DIR Trend by Grade")
    plt.xlabel("Evaluation Date")
    plt.ylabel("DIR")
    plt.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "trend_dir_by_grade.png"), dpi=300)
    plt.close()


def plot_score_distributions_dyslexic_only(df):
    ddf = df[df["is_dyslexic"] == True].copy()
    if ddf.empty:
        return

    long_df = pd.DataFrame(
        {
            "grade": np.concatenate([ddf["grade"].values, ddf["grade"].values]),
            "phase": np.concatenate(
                [np.array(["Before"] * len(ddf)), np.array(["After"] * len(ddf))]
            ),
            "score_100": np.concatenate([ddf["orig_100"].values, ddf["adj_100"].values]),
        }
    )
    long_df.to_csv(
        os.path.join(OUT_DIR, "score_distribution_dyslexic_only_long.csv"), index=False
    )

    # Boxplot
    plt.figure(figsize=(9.5, 5.5))
    sns.boxplot(data=long_df, x="phase", y="score_100")
    plt.title("Dyslexic Students: Score Distribution Before vs After Mitigation")
    plt.xlabel("Phase")
    plt.ylabel("Score (0-100)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "score_distribution_dyslexic_only_boxplot.png"), dpi=300)
    plt.close()

    # Histogram
    g = sns.displot(
        data=long_df,
        x="score_100",
        hue="phase",
        kind="hist",
        stat="density",
        common_norm=False,
        bins=25,
        height=4.2,
        aspect=1.35,
    )
    g.fig.suptitle("Dyslexic Students: Score Histograms Before vs After", y=1.03)
    g.savefig(
        os.path.join(OUT_DIR, "score_distribution_dyslexic_only_histograms.png"),
        dpi=300,
    )
    plt.close(g.fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    db = get_db()

    user_df = fetch_user_images_df(db)
    if user_df.empty:
        print("No score data found in userImages.")
        return

    spd_dir_tbl = make_spd_dir_before_after(user_df)
    plot_spd_dir_before_after(spd_dir_tbl)
    plot_mean_before_after_dyslexic_only(user_df)

    fairness_df = fetch_fairness_reports_df(db)
    plot_spd_dir_trend(fairness_df)

    plot_score_distributions_dyslexic_only(user_df)

    print("Done. Outputs written to:", OUT_DIR)
    print("- spd_dir_before_after_by_grade.csv/.png")
    print("- mean_before_after_dyslexic_only.png")
    print("- mean_scores_before_after_dyslexic_only.csv")
    print("- trend_spd_by_grade.png")
    print("- trend_dir_by_grade.png")
    print("- score_distribution_dyslexic_only_boxplot.png")
    print("- score_distribution_dyslexic_only_histograms.png")


if __name__ == "__main__":
    main()
