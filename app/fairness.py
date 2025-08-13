from .schemas import FairnessReport

def empty_fairness_report() -> FairnessReport:
    # Phase 1 placeholders (real metrics come in Phase 2)
    return FairnessReport(spd=0.0, dir=1.0, eod=0.0, mitigation_used=None)
