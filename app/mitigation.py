"""
Bias-Aware Scoring Engine: Conditional Post-Processing Fairness Mitigation
===========================================================================

Specification Implementation:
-----------------------------
This module implements a conditional, post-processing bias mitigation mechanism
to address residual scoring bias against dyslexic students arising from 
historically biased human grading practices.

Key Design Principles:
1. CONDITIONAL TRIGGERING - Only when fairness thresholds are violated
2. GRADE-AWARE CALIBRATION - Different calibration per grade level
3. PROPORTIONAL & BOUNDED - Prevents over-correction
4. NON-DYSLEXIC SCORES UNCHANGED - Only adjusts protected group
5. NO MODEL MODIFICATION - Post-processing only
6. FULL TRANSPARENCY - All actions logged and auditable

Fairness Metrics:
- SPD (Statistical Parity Difference): Difference in positive outcome rates
- DIR (Disparate Impact Ratio): Ratio of positive outcome rates
- EOD (Equal Opportunity Difference): Requires ground truth labels

Thresholds (configurable):
- |SPD| > ε (default ε = 0.1)
- DIR < 0.8 (80% rule from EEOC guidelines)

Academic Alignment:
- AIF360 (IBM Fairness 360) methodology
- Equalized Odds post-processing (Hardt et al., 2016)
- Calibrated Equalized Odds (Pleiss et al., 2017)

Author: Nuwan (Bias-Aware Scoring Engine)
Date: 2026-01-22
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


@dataclass
class MitigationRecord:
    """
    Transparency record for each mitigation action.
    Enables auditability by educators and researchers.
    """
    timestamp: str
    grade: int
    protected_attribute: str
    protected_value: bool  # dyslexic_flag value
    
    # Scores
    original_score: float
    adjusted_score: float
    multiplier_applied: float  # e.g., 1.08 (8% boost)
    absolute_boost: float      # e.g., +4.5 points
    
    # Fairness metrics at time of decision
    spd_value: float
    dir_value: float
    eod_value: Optional[float]
    
    # Threshold violations
    spd_threshold_violated: bool
    dir_threshold_violated: bool
    
    # Calibration details
    calibration_method: str
    calibration_source: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/logging."""
        return {
            "timestamp": self.timestamp,
            "grade": self.grade,
            "protected_attribute": self.protected_attribute,
            "protected_value": self.protected_value,
            "original_score": self.original_score,
            "adjusted_score": self.adjusted_score,
            "multiplier_applied": self.multiplier_applied,
            "absolute_boost": self.absolute_boost,
            "fairness_metrics": {
                "spd": self.spd_value,
                "dir": self.dir_value,
                "eod": self.eod_value
            },
            "threshold_violations": {
                "spd_violated": self.spd_threshold_violated,
                "dir_violated": self.dir_threshold_violated
            },
            "calibration": {
                "method": self.calibration_method,
                "source": self.calibration_source
            }
        }


@dataclass
class GradeFairnessMetrics:
    """Fairness metrics for a specific grade level."""
    grade: int
    spd: float
    dir: float
    eod: Optional[float] = None
    mean_dyslexic: float = 0.0
    mean_non_dyslexic: float = 0.0
    n_dyslexic: int = 0
    n_non_dyslexic: int = 0
    sample_size: int = 0
    calibration_multiplier: Optional[float] = None
    evaluated_at: Optional[str] = None


class ConditionalFairnessMitigator:
    """
    Conditional Post-Processing Bias Mitigation.
    
    Implements the specification:
    - Bias mitigation triggered ONLY when thresholds are violated
    - Grade-aware calibrated adjustment
    - Proportional, bounded corrections
    - Full transparency and auditability
    """
    
    # ===============================
    # CONFIGURABLE THRESHOLDS
    # ===============================
    
    # Statistical Parity Difference threshold
    # Mitigation triggered when SPD < -SPD_EPSILON (dyslexic scoring LOWER)
    # Note: We only care about UNFAVORABLE bias (dyslexic disadvantaged)
    SPD_EPSILON: float = 0.1  # On 0-1 scale (10% difference)
    
    # Disparate Impact Ratio threshold (80% rule)
    # Mitigation triggered ONLY when DIR < 0.8 (dyslexic disadvantaged)
    # We do NOT reduce scores if DIR > 1.25 (dyslexic advantaged)
    DIR_MIN: float = 0.8   # EEOC 80% rule - below this = unfavorable bias
    
    # Maximum allowed multiplier (prevents over-correction)
    # 1.15 = Max 15% proportional boost
    MAX_MULTIPLIER: float = 1.15
    
    # Minimum samples required to apply mitigation
    MIN_SAMPLES_FOR_MITIGATION: int = 10
    
    def __init__(self):
        """Initialize the mitigator with empty state."""
        # Current fairness metrics per grade (from Fairness Dashboard)
        self.grade_metrics: Dict[int, GradeFairnessMetrics] = {}
        
        # Calibration multipliers per grade (calculated from metrics)
        # Default is 1.0 (no change)
        self.calibration_multipliers: Dict[int, float] = {
            grade: 1.0 for grade in range(3, 9)
        }
        
        # Transparency log
        self.mitigation_log: List[MitigationRecord] = []
        
        # Whether mitigation is active per grade
        self.mitigation_active: Dict[int, bool] = {
            grade: False for grade in range(3, 9)
        }
    
    def update_fairness_metrics(self, grade: int, metrics: Dict[str, Any]) -> None:
        """
        Update fairness metrics for a grade from Fairness Dashboard.
        
        This should be called after running the fairness evaluation script.
        
        Args:
            grade: Grade level (3-8)
            metrics: Dictionary with spd, dir, mean_dyslexic, mean_non_dyslexic, etc.
        """
        self.grade_metrics[grade] = GradeFairnessMetrics(
            grade=grade,
            spd=metrics.get("spd", 0.0),
            dir=metrics.get("dir", 1.0),
            eod=metrics.get("eod"),
            mean_dyslexic=metrics.get("mean_dyslexic", 0.0),
            mean_non_dyslexic=metrics.get("mean_non_dyslexic", 0.0),
            n_dyslexic=metrics.get("n_dyslexic", 0),
            n_non_dyslexic=metrics.get("n_non_dyslexic", 0),
            sample_size=metrics.get("sample_size", 0),
            calibration_multiplier=metrics.get("calibration_multiplier"),
            evaluated_at=metrics.get("evaluated_at", datetime.utcnow().isoformat())
        )
        
        # Check if thresholds are violated and update calibration
        self._update_calibration_for_grade(grade)
    
    def _check_threshold_violations(self, grade: int) -> tuple:
        """
        Check if UNFAVORABLE bias thresholds are violated for a grade.
        
        Key Principle: We only mitigate when dyslexic students are DISADVANTAGED.
        - If dyslexic students score LOWER (DIR < 0.8): Boost their scores
        - If dyslexic students score EQUAL or HIGHER: No adjustment
        
        This ensures dyslexic students are graded like normal students,
        but protected from unfair penalties due to biased training data.
        
        Returns:
            Tuple of (spd_violated, dir_violated, should_mitigate)
        """
        metrics = self.grade_metrics.get(grade)
        
        if metrics is None:
            return False, False, False
        
        # Check sample size requirement
        if metrics.sample_size < self.MIN_SAMPLES_FOR_MITIGATION:
            return False, False, False
        
        # Check SPD threshold: SPD < -ε (dyslexic scoring LOWER than non-dyslexic)
        # Negative SPD means dyslexic students are disadvantaged
        spd_violated = metrics.spd < -self.SPD_EPSILON
        
        # Check DIR threshold: DIR < 0.8 (dyslexic disadvantaged)
        # We do NOT trigger for DIR > 1.25 anymore - that's favorable bias
        dir_violated = metrics.dir < self.DIR_MIN
        
        # Mitigation triggered ONLY if dyslexic students are DISADVANTAGED
        should_mitigate = spd_violated or dir_violated
        
        return spd_violated, dir_violated, should_mitigate
    
    def _update_calibration_for_grade(self, grade: int) -> None:
        """
        Calculate and update calibration multiplier for a grade.
        
        Smarter Logic (Proportional Scaling):
        Instead of adding a flat number, we calculate a multiplier.
        This ensures that higher-quality essays get a larger absolute boost 
        than poor-quality (short) essays, preserving academic merit.
        
        multiplier = Mean(Non-Dyslexic) / Mean(Dyslexic)
        """
        metrics = self.grade_metrics.get(grade)
        
        if metrics is None:
            self.calibration_multipliers[grade] = 1.0
            self.mitigation_active[grade] = False
            return
        
        spd_violated, dir_violated, should_mitigate = self._check_threshold_violations(grade)
        
        if not should_mitigate:
            # No unfavorable bias detected - score like a normal student
            self.calibration_multipliers[grade] = 1.0
            self.mitigation_active[grade] = False
            print(f"[MITIGATION] Grade {grade}: No unfavorable bias (SPD={metrics.spd:.3f}, DIR={metrics.dir:.3f})")
            print(f"             Dyslexic students will be scored like normal students.")
            return
        
        # Calculate raw multiplier (how much higher non-dyslexics score proportionally)
        # First check if Firebase already calculated the multiplier
        if hasattr(metrics, 'calibration_multiplier') and metrics.calibration_multiplier and metrics.calibration_multiplier > 1.0:
            bounded_multiplier = min(metrics.calibration_multiplier, self.MAX_MULTIPLIER)
            print(f"             Using Firebase calibration_multiplier: {bounded_multiplier:.3f}")
        else:
            # Calculate from means - use a safety floor to avoid DivisionByZero
            mean_d = max(1.0, metrics.mean_dyslexic)
            mean_nd = metrics.mean_non_dyslexic if metrics.mean_non_dyslexic > 0 else mean_d
            raw_multiplier = mean_nd / mean_d
            
            # Debug output
            print(f"             Calculating multiplier: {mean_nd}/{mean_d} = {raw_multiplier:.3f}")
            
            # IMPORTANT: Only apply BOOST (multiplier > 1.0), never reduce.
            # Bound the multiplier to MAX_MULTIPLIER (e.g., 15% cap)
            bounded_multiplier = max(1.0, min(raw_multiplier, self.MAX_MULTIPLIER))
        
        self.calibration_multipliers[grade] = round(bounded_multiplier, 3)
        self.mitigation_active[grade] = True
        
        violation_type = []
        if spd_violated:
            violation_type.append(f"SPD={metrics.spd:.3f}")
        if dir_violated:
            violation_type.append(f"DIR={metrics.dir:.3f}")
        
        print(f"[MITIGATION] Grade {grade}: UNFAVORABLE BIAS DETECTED ({', '.join(violation_type)})")
        print(f"             Applying Proportional Boost: x{bounded_multiplier:.3f} (+{(bounded_multiplier-1)*100:.1f}%)")
    
    def transform(self, raw_score: float, dyslexic_flag: bool, grade: int) -> tuple:
        """
        Apply SMARTER conditional fairness transformation.
        
        New Logic: Proportional scaling instead of flat addition.
        This protects against giving large boosts to low-effort (short) essays.
        """
        # RULE 1: Non-dyslexic scores are NEVER adjusted
        if not dyslexic_flag:
            return raw_score, None
        
        # RULE 2: Check if mitigation is active for grade
        if not self.mitigation_active.get(grade, False):
            return raw_score, None
        
        # RULE 3: Apply proportional multiplier
        metrics = self.grade_metrics.get(grade)
        multiplier = self.calibration_multipliers.get(grade, 1.0)
        
        if multiplier <= 1.001:
            return raw_score, None
        
        # Apply adjustment proportionally
        adjusted_score = raw_score * multiplier
        
        # Clamp to valid range
        adjusted_score = max(0.0, min(100.0, adjusted_score))
        absolute_boost = adjusted_score - raw_score
        
        # RULE 4: Create transparency record
        spd_violated, dir_violated, _ = self._check_threshold_violations(grade)
        
        record = MitigationRecord(
            timestamp=datetime.utcnow().isoformat(),
            grade=grade,
            protected_attribute="dyslexic_flag",
            protected_value=True,
            original_score=raw_score,
            adjusted_score=adjusted_score,
            multiplier_applied=multiplier,
            absolute_boost=absolute_boost,
            spd_value=metrics.spd if metrics else 0.0,
            dir_value=metrics.dir if metrics else 1.0,
            eod_value=metrics.eod if metrics else None,
            spd_threshold_violated=spd_violated,
            dir_threshold_violated=dir_violated,
            calibration_method="Proportional Grade-Aware Multiplier",
            calibration_source="Fairness Dashboard (Firebase)"
        )
        
        # Add to log
        self.mitigation_log.append(record)
        
        # Print for transparency
        print(f"[MITIGATION APPLIED] Grade {grade} (Proportional Scaling)")
        print(f"   Original: {raw_score:.2f} -> Adjusted: {adjusted_score:.2f} (Boost: +{absolute_boost:.2f})")
        print(f"   Factor: x{multiplier:.3f} (+{(multiplier-1)*100:.1f}%)")
        
        return adjusted_score, record
    
    def get_mitigation_status(self) -> Dict[str, Any]:
        """
        Get current mitigation status for all grades.
        """
        status = {}
        for grade in range(3, 9):
            metrics = self.grade_metrics.get(grade)
            spd_violated, dir_violated, should_mitigate = self._check_threshold_violations(grade)
            
            status[f"grade_{grade}"] = {
                "mitigation_active": self.mitigation_active.get(grade, False),
                "calibration_multiplier": self.calibration_multipliers.get(grade, 1.0),
                "thresholds_violated": should_mitigate,
                "spd_violated": spd_violated,
                "dir_violated": dir_violated,
                "metrics": {
                    "spd": metrics.spd if metrics else None,
                    "dir": metrics.dir if metrics else None,
                    "sample_size": metrics.sample_size if metrics else 0
                }
            }
        return status
    
    def get_transparency_report(self) -> Dict[str, Any]:
        """
        Generate full transparency report for auditability.
        """
        return {
            "mitigation_specification": {
                "approach": "Conditional Post-Processing Bias Mitigation",
                "trigger_conditions": {
                    "spd_threshold": f"SPD < -{self.SPD_EPSILON}",
                    "dir_threshold": f"DIR < {self.DIR_MIN}"
                },
                "adjustment_method": "Proportional Grade-Aware Multiplier",
                "max_multiplier": f"x{self.MAX_MULTIPLIER} (+15%)",
                "min_samples_required": self.MIN_SAMPLES_FOR_MITIGATION,
                "protected_attribute": "dyslexic_flag",
                "non_dyslexic_adjustment": "NONE (unchanged)"
            },
            "current_status": self.get_mitigation_status(),
            "calibration_multipliers": self.calibration_multipliers,
            "total_mitigations_applied": len(self.mitigation_log),
            "academic_alignment": [
                "AIF360 (IBM Fairness 360)",
                "Equalized Odds (Hardt et al., 2016)",
                "Calibrated Equalized Odds (Pleiss et al., 2017)",
                "80% Rule (EEOC Disparate Impact Guidelines)"
            ],
            "transparency_log_count": len(self.mitigation_log)
        }
    
    def export_mitigation_log(self) -> List[Dict]:
        """Export all mitigation records for research documentation."""
        return [record.to_dict() for record in self.mitigation_log]
    
    def load_metrics_from_firebase(self) -> None:
        """
        Load the latest fairness metrics from Firebase on startup.
        This ensures the mitigator is initialized with the calculated bias data.
        
        If Firebase credentials are not available (e.g., in Docker without the key file),
        the mitigator will remain inactive and all scores will pass through unchanged.
        """
        import os
        
        # Check if credentials file exists before attempting to load
        if not os.path.exists("serviceAccountKey.json"):
            print("[MITIGATION] serviceAccountKey.json not found - mitigation will remain inactive.")
            print("[MITIGATION] This is expected in production Cloud Run (uses Workload Identity).")
            return
        
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
            
            # Initialize Firebase if not already done
            try:
                cred = credentials.Certificate("serviceAccountKey.json")
                firebase_admin.initialize_app(cred)
            except ValueError:
                # Already initialized
                pass
            except Exception as e:
                print(f"[MITIGATION] Failed to initialize Firebase: {e}")
                return
            
            db = firestore.client()
            
            # Fetch ALL fairness reports and filter in Python (avoids needing composite index)
            all_docs = list(db.collection("fairnessReports").stream())
            
            # Group by grade and get latest for each (using doc ID which has date)
            grade_reports = {}
            for doc in all_docs:
                report = doc.to_dict()
                grade = report.get("grade")
                doc_id = doc.id  # e.g., "grade_7_20260122"
                
                if grade is not None:
                    # Keep the latest report per grade (doc ID contains date, so alphabetically larger = newer)
                    if grade not in grade_reports:
                        grade_reports[grade] = (doc_id, report)
                    else:
                        existing_id = grade_reports[grade][0]
                        # Compare doc IDs - later dates have higher values
                        if doc_id > existing_id:
                            grade_reports[grade] = (doc_id, report)
            
            # Load each grade's metrics
            for grade, (doc_id, report) in grade_reports.items():
                print(f"[MITIGATION] Loading Grade {grade} metrics from {doc_id}...")
                self.update_fairness_metrics(grade, report)
            
            # Print summary
            active_grades = [g for g, active in self.mitigation_active.items() if active]
            if active_grades:
                print(f"[MITIGATION] Loaded! Active mitigation for grades: {active_grades}")
            else:
                print(f"[MITIGATION] Loaded! No unfavorable bias detected - all grades will score normally.")
                
        except Exception as e:
            print(f"[MITIGATION] Warning: Could not load metrics from Firebase: {e}")
            print(f"[MITIGATION] Mitigation will remain inactive until metrics are loaded.")


# ========================
# Singleton Instance
# ========================
mitigator = ConditionalFairnessMitigator()

# Auto-load metrics from Firebase on module import
mitigator.load_metrics_from_firebase()
