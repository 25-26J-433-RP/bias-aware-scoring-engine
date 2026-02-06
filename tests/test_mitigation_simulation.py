"""
Test Script: Simulate Unfavorable Bias and Test Mitigation
==========================================================

This script simulates a scenario where dyslexic students are scoring 
LOWER than non-dyslexic students (unfavorable bias), and verifies 
that the mitigation system correctly applies a boost.

Run this to test the conditional mitigation logic quickly.
"""

import sys
sys.path.insert(0, '.')

from app.mitigation import mitigator, ConditionalFairnessMitigator


def test_mitigation():
    print("=" * 70)
    print("TESTING CONDITIONAL BIAS MITIGATION")
    print("=" * 70)
    
    # Reset the mitigator for fresh testing
    fresh_mitigator = ConditionalFairnessMitigator()
    
    # =========================================
    # TEST 1: Simulate UNFAVORABLE bias (DIR < 0.8)
    # Dyslexic students scoring LOWER
    # =========================================
    print("\n" + "-" * 70)
    print("TEST 1: Unfavorable Bias (DIR < 0.8)")
    print("Scenario: Dyslexic students scoring 10 points LOWER on average")
    print("-" * 70)
    
    # Simulate metrics where dyslexic students score lower
    unfavorable_metrics = {
        "spd": -0.15,  # Negative = dyslexic disadvantaged
        "dir": 0.75,   # Below 0.8 = unfavorable bias
        "mean_dyslexic": 55.0,      # Dyslexic average
        "mean_non_dyslexic": 65.0,  # Non-dyslexic average (10 points higher)
        "n_dyslexic": 20,
        "n_non_dyslexic": 30,
        "sample_size": 50,
    }
    
    # Update mitigator with these metrics for Grade 5
    fresh_mitigator.update_fairness_metrics(grade=5, metrics=unfavorable_metrics)
    
    # Test scoring a dyslexic student
    print("\nScoring a dyslexic student essay (raw score: 55.0)...")
    adjusted_score, record = fresh_mitigator.transform(
        raw_score=55.0,
        dyslexic_flag=True,
        grade=5
    )
    
    print(f"\nRESULT:")
    print(f"   Original Score: 55.0")
    print(f"   Adjusted Score: {adjusted_score:.2f}")
    print(f"   Boost Applied:  +{adjusted_score - 55.0:.2f}")
    print(f"   Mitigation Record: {'Created' if record else 'None'}")
    
    if record:
        print(f"\n   Transparency Details:")
        print(f"      SPD Violated: {record.spd_threshold_violated}")
        print(f"      DIR Violated: {record.dir_threshold_violated}")
        print(f"      SPD Value: {record.spd_value}")
        print(f"      DIR Value: {record.dir_value}")
    
    assert adjusted_score > 55.0, "ERROR: Score should be boosted!"
    print("\n   [PASS] TEST 1: Dyslexic score correctly boosted!")
    
    # =========================================
    # TEST 2: Simulate NO bias (0.8 <= DIR <= 1.25)
    # Both groups scoring equally
    # =========================================
    print("\n" + "-" * 70)
    print("TEST 2: No Significant Bias (0.8 <= DIR <= 1.25)")
    print("Scenario: Both groups scoring similarly")
    print("-" * 70)
    
    fair_metrics = {
        "spd": 0.02,   # Near zero = fair
        "dir": 1.0,    # 1.0 = perfect parity
        "mean_dyslexic": 62.0,
        "mean_non_dyslexic": 63.0,  # Only 1 point difference
        "n_dyslexic": 25,
        "n_non_dyslexic": 35,
        "sample_size": 60,
    }
    
    fresh_mitigator.update_fairness_metrics(grade=6, metrics=fair_metrics)
    
    print("\nScoring a dyslexic student essay (raw score: 62.0)...")
    adjusted_score, record = fresh_mitigator.transform(
        raw_score=62.0,
        dyslexic_flag=True,
        grade=6
    )
    
    print(f"\nRESULT:")
    print(f"   Original Score: 62.0")
    print(f"   Adjusted Score: {adjusted_score:.2f}")
    print(f"   Change:         {adjusted_score - 62.0:+.2f}")
    print(f"   Mitigation Record: {'Created' if record else 'None (no adjustment)'}")
    
    assert adjusted_score == 62.0, "ERROR: Score should NOT be changed!"
    print("\n   [PASS] TEST 2: No adjustment when no unfavorable bias!")
    
    # =========================================
    # TEST 3: Simulate FAVORABLE bias (DIR > 1.25)
    # Dyslexic students scoring HIGHER
    # =========================================
    print("\n" + "-" * 70)
    print("TEST 3: Favorable Bias (DIR > 1.25)")
    print("Scenario: Dyslexic students scoring HIGHER (like your current data)")
    print("-" * 70)
    
    favorable_metrics = {
        "spd": 0.45,   # Positive = dyslexic advantaged
        "dir": 3.0,    # Way above 1.25 = favorable bias
        "mean_dyslexic": 75.0,      # Dyslexic scoring higher
        "mean_non_dyslexic": 60.0,  # Non-dyslexic lower
        "n_dyslexic": 15,
        "n_non_dyslexic": 40,
        "sample_size": 55,
    }
    
    fresh_mitigator.update_fairness_metrics(grade=7, metrics=favorable_metrics)
    
    print("\nScoring a dyslexic student essay (raw score: 75.0)...")
    adjusted_score, record = fresh_mitigator.transform(
        raw_score=75.0,
        dyslexic_flag=True,
        grade=7
    )
    
    print(f"\nRESULT:")
    print(f"   Original Score: 75.0")
    print(f"   Adjusted Score: {adjusted_score:.2f}")
    print(f"   Change:         {adjusted_score - 75.0:+.2f}")
    print(f"   Mitigation Record: {'Created' if record else 'None (no adjustment)'}")
    
    assert adjusted_score == 75.0, "ERROR: Score should NOT be reduced!"
    print("\n   [PASS] TEST 3: Score NOT reduced even with favorable bias!")
    
    # =========================================
    # TEST 4: Non-dyslexic student (never adjusted)
    # =========================================
    print("\n" + "-" * 70)
    print("TEST 4: Non-Dyslexic Student")
    print("Scenario: Non-dyslexic student should NEVER be adjusted")
    print("-" * 70)
    
    print("\nScoring a NON-dyslexic student essay (raw score: 55.0)...")
    adjusted_score, record = fresh_mitigator.transform(
        raw_score=55.0,
        dyslexic_flag=False,  # Non-dyslexic
        grade=5  # Same grade with unfavorable bias
    )
    
    print(f"\nRESULT:")
    print(f"   Original Score: 55.0")
    print(f"   Adjusted Score: {adjusted_score:.2f}")
    print(f"   Change:         {adjusted_score - 55.0:+.2f}")
    
    assert adjusted_score == 55.0, "ERROR: Non-dyslexic score should NEVER change!"
    print("\n   [PASS] TEST 4: Non-dyslexic score unchanged!")
    
    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    [PASS] TEST 1: Unfavorable bias (DIR < 0.8) -> Dyslexic score BOOSTED
    [PASS] TEST 2: No bias (0.8 <= DIR <= 1.25) -> Score UNCHANGED
    [PASS] TEST 3: Favorable bias (DIR > 1.25) -> Score UNCHANGED (never reduced)
    [PASS] TEST 4: Non-dyslexic student -> Score NEVER changed
    
    ALL TESTS PASSED!
    
    The mitigation system correctly:
    - Only boosts dyslexic scores when unfavorable bias exists
    - Never reduces any scores
    - Treats non-dyslexic students consistently
    """)
    
    # Show transparency report
    print("\n" + "-" * 70)
    print("MITIGATION STATUS BY GRADE")
    print("-" * 70)
    status = fresh_mitigator.get_mitigation_status()
    for grade_key, info in status.items():
        grade = grade_key.replace("grade_", "")
        active = "ACTIVE" if info["mitigation_active"] else "INACTIVE"
        multiplier = info["calibration_multiplier"]
        boost_pct = (multiplier - 1.0) * 100 if multiplier > 1.0 else 0.0
        print(f"   Grade {grade}: {active} (multiplier: x{multiplier:.3f}, +{boost_pct:.1f}%)")


if __name__ == "__main__":
    test_mitigation()
