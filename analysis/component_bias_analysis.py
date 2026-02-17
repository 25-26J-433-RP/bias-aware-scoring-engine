"""
Component-Level Bias Analysis Script

This script performs deep analysis of bias sources in the essay scoring system:
1. Component-wise score gaps (richness, organization, technical)
2. Penalty pattern analysis (word count, theme, technical violations)
3. ML vs Rule-based bias attribution
4. Statistical significance testing for each component
5. Targeted mitigation recommendations

Usage:
    python -m analysis.component_bias_analysis

Output:
    - Console: Detailed analysis per grade
    - File: component_bias_report.json
"""

import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
from scipy import stats
import json
from collections import defaultdict
from datetime import datetime


# Initialize Firestore (reuse if already initialized)
try:
    db = firestore.client()
except ValueError:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()


def fetch_essays_with_rubric(grade_filter: int):
    """Fetch essays with full rubric breakdown"""
    docs = db.collection("userImages").stream()
    rows = []

    for doc in docs:
        d = doc.to_dict()

        # Required fields
        if "score" not in d or "rubric" not in d:
            continue

        # Grade parsing
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
        fairness_report = rubric.get("fairness_report", {})
        rubric_notes = fairness_report.get("rubric_notes", {})

        rows.append({
            "doc_id": doc.id,
            "score": d["score"],
            "dyslexic_flag": details.get("dyslexic_flag", False),
            "grade": grade,
            # Component scores
            "richness_5": rubric.get("richness_5", 0),
            "organization_6": rubric.get("organization_6", 0),
            "technical_3": rubric.get("technical_3", 0),
            "total_14": rubric.get("total_14", 0),
            # Penalty information
            "word_count": rubric_notes.get("word_count", 0),
            "word_count_penalty": rubric_notes.get("word_count_penalty", 0),
            "theme_penalty": rubric_notes.get("theme_penalty", 0),
            "technical_penalty": rubric_notes.get("technical_penalty", 0),
            "theme_relevance": rubric_notes.get("theme_relevance", 1.0),
            "technical_violations": len(rubric_notes.get("technical_violations", [])),
            "grammar_issues": len(rubric_notes.get("grammar_issues", [])),
        })

    return pd.DataFrame(rows)


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = stats.tvar(group1), stats.tvar(group2)
    pooled_std = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
    return (group1.mean() - group2.mean()) / (pooled_std ** 0.5) if pooled_std > 0 else 0


def analyze_component(df, component_name, dyslexic_scores, non_dyslexic_scores):
    """Analyze a single component with statistical testing"""
    if len(dyslexic_scores) < 2 or len(non_dyslexic_scores) < 2:
        return {
            "component": component_name,
            "mean_dyslexic": 0,
            "mean_non_dyslexic": 0,
            "difference": 0,
            "t_statistic": 0,
            "p_value": 1.0,
            "cohens_d": 0,
            "effect_size": "insufficient_data",
            "significant": False,
        }

    t_stat, p_val = stats.ttest_ind(non_dyslexic_scores, dyslexic_scores)
    effect = cohens_d(non_dyslexic_scores, dyslexic_scores)

    if abs(effect) < 0.2:
        effect_label = "negligible"
    elif abs(effect) < 0.5:
        effect_label = "small"
    elif abs(effect) < 0.8:
        effect_label = "medium"
    else:
        effect_label = "large"

    return {
        "component": component_name,
        "mean_dyslexic": round(dyslexic_scores.mean(), 3),
        "mean_non_dyslexic": round(non_dyslexic_scores.mean(), 3),
        "difference": round(non_dyslexic_scores.mean() - dyslexic_scores.mean(), 3),
        "t_statistic": round(t_stat, 3),
        "p_value": round(p_val, 4),
        "cohens_d": round(effect, 3),
        "effect_size": effect_label,
        "significant": p_val < 0.05,
    }


def analyze_penalties(df_dys, df_non_dys):
    """Analyze which penalties are triggered more for dyslexic students"""
    results = {}

    # Word count analysis
    results["word_count"] = {
        "mean_dyslexic": round(df_dys["word_count"].mean(), 1),
        "mean_non_dyslexic": round(df_non_dys["word_count"].mean(), 1),
        "penalty_rate_dyslexic": round((df_dys["word_count_penalty"] > 0).mean(), 3),
        "penalty_rate_non_dyslexic": round((df_non_dys["word_count_penalty"] > 0).mean(), 3),
        "avg_penalty_dyslexic": round(df_dys["word_count_penalty"].mean(), 3),
        "avg_penalty_non_dyslexic": round(df_non_dys["word_count_penalty"].mean(), 3),
    }

    # Theme relevance analysis
    results["theme"] = {
        "mean_relevance_dyslexic": round(df_dys["theme_relevance"].mean(), 3),
        "mean_relevance_non_dyslexic": round(df_non_dys["theme_relevance"].mean(), 3),
        "penalty_rate_dyslexic": round((df_dys["theme_penalty"] > 0).mean(), 3),
        "penalty_rate_non_dyslexic": round((df_non_dys["theme_penalty"] > 0).mean(), 3),
        "avg_penalty_dyslexic": round(df_dys["theme_penalty"].mean(), 3),
        "avg_penalty_non_dyslexic": round(df_non_dys["theme_penalty"].mean(), 3),
    }

    # Technical violations analysis
    results["technical"] = {
        "violations_per_essay_dyslexic": round(df_dys["technical_violations"].mean(), 2),
        "violations_per_essay_non_dyslexic": round(df_non_dys["technical_violations"].mean(), 2),
        "grammar_issues_dyslexic": round(df_dys["grammar_issues"].mean(), 2),
        "grammar_issues_non_dyslexic": round(df_non_dys["grammar_issues"].mean(), 2),
        "avg_penalty_dyslexic": round(df_dys["technical_penalty"].mean(), 3),
        "avg_penalty_non_dyslexic": round(df_non_dys["technical_penalty"].mean(), 3),
    }

    return results


def recommend_mitigation(component_analysis, penalty_analysis):
    """Generate targeted mitigation recommendations"""
    recommendations = []

    # Check component-level bias
    for comp in component_analysis:
        if comp["significant"] and abs(comp["cohens_d"]) > 0.5:
            recommendations.append({
                "type": "component_bias",
                "component": comp["component"],
                "severity": comp["effect_size"],
                "action": f"Investigate {comp['component']} scoring - large bias detected",
                "suggested_fix": f"Consider recalibrating {comp['component']} or reviewing ML model training for this dimension"
            })

    # Check penalty bias
    if penalty_analysis["word_count"]["penalty_rate_dyslexic"] > penalty_analysis["word_count"]["penalty_rate_non_dyslexic"] * 1.5:
        recommendations.append({
            "type": "rule_bias",
            "component": "word_count",
            "severity": "medium",
            "action": "Dyslexic students hit word count penalties more often",
            "suggested_fix": "Consider lowering word count threshold or making it grade-specific"
        })

    if penalty_analysis["theme"]["penalty_rate_dyslexic"] > penalty_analysis["theme"]["penalty_rate_non_dyslexic"] * 1.5:
        recommendations.append({
            "type": "rule_bias",
            "component": "theme_relevance",
            "severity": "medium",
            "action": "Dyslexic students receive more theme penalties",
            "suggested_fix": "Review theme detection algorithm - may be sensitive to writing patterns"
        })

    if penalty_analysis["technical"]["avg_penalty_dyslexic"] > penalty_analysis["technical"]["avg_penalty_non_dyslexic"] * 1.5:
        recommendations.append({
            "type": "rule_bias",
            "component": "technical",
            "severity": "high",
            "action": "Technical penalties disproportionately affect dyslexic students",
            "suggested_fix": "Consider relaxing punctuation/grammar rules for dyslexic students or excluding from mitigation"
        })

    # Overall recommendation
    if not recommendations:
        recommendations.append({
            "type": "no_bias",
            "severity": "none",
            "action": "No significant bias detected in components or penalties",
            "suggested_fix": "Current scoring appears fair - monitor over time"
        })

    return recommendations


def run_component_analysis():
    """Main analysis function"""
    print("\n[COMPONENT BIAS ANALYSIS] Starting comprehensive analysis...")
    print("=" * 70)

    all_results = {}

    for grade in range(3, 9):
        print(f"\n{'='*70}")
        print(f"GRADE {grade} ANALYSIS")
        print(f"{'='*70}")

        df = fetch_essays_with_rubric(grade_filter=grade)

        if df.empty or len(df) < 10:
            print(f"[WARNING] Insufficient data for Grade {grade} (n={len(df)})")
            continue

        df_dys = df[df["dyslexic_flag"] == True]
        df_non_dys = df[df["dyslexic_flag"] == False]

        if len(df_dys) < 5 or len(df_non_dys) < 5:
            print(f"[WARNING] Insufficient group sizes (Dys={len(df_dys)}, Non-Dys={len(df_non_dys)})")
            continue

        print(f"\nSample sizes: Dyslexic={len(df_dys)}, Non-Dyslexic={len(df_non_dys)}")

        # Component analysis
        print("\n--- COMPONENT-LEVEL ANALYSIS ---")
        component_results = []

        for component in ["richness_5", "organization_6", "technical_3", "total_14"]:
            result = analyze_component(
                df, component,
                df_dys[component], df_non_dys[component]
            )
            component_results.append(result)

            status = "[BIAS DETECTED]" if result["significant"] and abs(result["cohens_d"]) > 0.5 else "[OK]"
            print(f"\n{status} {component}:")
            print(f"  Mean Dys: {result['mean_dyslexic']:.2f}, Mean Non-Dys: {result['mean_non_dyslexic']:.2f}")
            print(f"  Difference: {result['difference']:.2f}, p-value: {result['p_value']:.4f}")
            print(f"  Effect size: {result['effect_size']} (d={result['cohens_d']:.3f})")

        # Penalty analysis
        print("\n--- PENALTY PATTERN ANALYSIS ---")
        penalty_results = analyze_penalties(df_dys, df_non_dys)

        print("\nWord Count Penalties:")
        print(f"  Dyslexic: {penalty_results['word_count']['avg_penalty_dyslexic']:.3f} avg penalty, "
              f"{penalty_results['word_count']['penalty_rate_dyslexic']*100:.1f}% trigger rate")
        print(f"  Non-Dyslexic: {penalty_results['word_count']['avg_penalty_non_dyslexic']:.3f} avg penalty, "
              f"{penalty_results['word_count']['penalty_rate_non_dyslexic']*100:.1f}% trigger rate")

        print("\nTheme Relevance Penalties:")
        print(f"  Dyslexic: {penalty_results['theme']['avg_penalty_dyslexic']:.3f} avg penalty, "
              f"{penalty_results['theme']['penalty_rate_dyslexic']*100:.1f}% trigger rate")
        print(f"  Non-Dyslexic: {penalty_results['theme']['avg_penalty_non_dyslexic']:.3f} avg penalty, "
              f"{penalty_results['theme']['penalty_rate_non_dyslexic']*100:.1f}% trigger rate")

        print("\nTechnical Violations:")
        print(f"  Dyslexic: {penalty_results['technical']['violations_per_essay_dyslexic']:.2f} violations/essay, "
              f"{penalty_results['technical']['avg_penalty_dyslexic']:.3f} avg penalty")
        print(f"  Non-Dyslexic: {penalty_results['technical']['violations_per_essay_non_dyslexic']:.2f} violations/essay, "
              f"{penalty_results['technical']['avg_penalty_non_dyslexic']:.3f} avg penalty")

        # Generate recommendations
        recommendations = recommend_mitigation(component_results, penalty_results)

        print("\n--- MITIGATION RECOMMENDATIONS ---")
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['type'].upper()}] {rec['action']}")
            print(f"   Severity: {rec['severity']}")
            print(f"   Suggested Fix: {rec['suggested_fix']}")

        # Store results
        all_results[f"grade_{grade}"] = {
            "sample_sizes": {"dyslexic": len(df_dys), "non_dyslexic": len(df_non_dys)},
            "component_analysis": component_results,
            "penalty_analysis": penalty_results,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
        }

    # Save to file
    output_file = "component_bias_report.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"[SUCCESS] Analysis complete. Results saved to {output_file}")
    print(f"{'='*70}\n")

    return all_results


if __name__ == "__main__":
    run_component_analysis()
