from app.fairness import demo_fairness_report

def test_demo_fairness_report_runs():
    fr = demo_fairness_report(n=100, seed=42)
    assert isinstance(fr.spd, float)
    assert isinstance(fr.dir, float)
    assert isinstance(fr.eod, float)
    # Sanity: values not all zero
    assert abs(fr.spd) > 0 or abs(fr.dir - 1) > 0
