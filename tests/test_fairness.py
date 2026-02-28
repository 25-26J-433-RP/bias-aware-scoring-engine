from app.fairness import demo_fairness_report

def test_demo_fairness_report_runs():
    fr = demo_fairness_report(n=100, seed=42)
    assert isinstance(fr.spd, float)
    assert isinstance(fr.dir, float)
    assert isinstance(fr.eod, float)
    # Sanity: metric values are bounded and computation is stable.
    assert -1.0 <= fr.spd <= 1.0
    assert fr.dir >= 0.0
