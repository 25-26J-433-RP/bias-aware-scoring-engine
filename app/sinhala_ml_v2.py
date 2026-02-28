import os
import torch
import threading
from typing import Optional

MODEL_SOURCE = "akura-official/xlm-roberta-large-sinhala-multihead"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Detect CI / test environment
IS_TEST = os.getenv("DISABLE_ML", "0") == "1"

# Lazy-loaded model and tokenizer (initialized on first use)
_tokenizer = None
_model = None
_load_lock = threading.Lock()


def load_model():
    """
    Lazy load both model and tokenizer on first use.
    This ensures NO loading happens at import time.
    """
    global _model, _tokenizer
    
    # Fast path: already loaded
    if _model is not None:
        return _model, _tokenizer
        
    # Thread-safe loading
    with _load_lock:
        # Double check after acquiring lock
        if _model is not None:
            return _model, _tokenizer
            
        if IS_TEST:
            return None, None
    
    print("[SINHALA-ML] Loading model and tokenizer...")
    
    # Import inside function to avoid any import-time side effects
    from transformers import AutoTokenizer
    from .model_multitask_xlmr import SinhalaMultiHeadRegressor
    
    # In production/deployment, we should prefer local files to avoid 429 errors
    # The Dockerfile pre-downloads these to /app/hf_cache
    is_offline = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
    
    try:
        print(f"[SINHALA-ML] Loading from cache (offline={is_offline})...")
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_SOURCE,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=is_offline
        )
        print("[SINHALA-ML] Tokenizer loaded.")
        
        _model = SinhalaMultiHeadRegressor.from_pretrained(
            MODEL_SOURCE,
            trust_remote_code=True,
            local_files_only=is_offline
        )
        _model.to(DEVICE)
        _model.eval()
        print("[SINHALA-ML] Model loaded successfully.")
        
    except Exception as e:
        print(f"[SINHALA-ML] Error loading model: {e}")
        # If we failed because of offline mode but files aren't there, 
        # that's a configuration error in the Dockerfile
        if is_offline:
            print("[SINHALA-ML] CRITICAL: Offline mode requested but files not found in cache!")
        raise
    
    return _model, _tokenizer


def _get_grade_adjustment_factor(grade: int, text_length: int) -> float:
    """
    Calculate a grade-level adjustment factor.
    Higher grades have stricter expectations, so scores are adjusted downward
    for simpler essays when evaluated at higher grades.
    
    Grade expectations (on 14-point scale):
    - Grade 3: expects ~6-8 points (more lenient)
    - Grade 4: expects ~7-9 points
    - Grade 5: expects ~8-10 points (middle)
    - Grade 6: expects ~9-11 points
    - Grade 7: expects ~10-12 points
    - Grade 8: expects ~11-13 points (strictest)
    
    Args:
        grade: Student grade level (3-8)
        text_length: Word count of essay
        
    Returns:
        Adjustment multiplier (< 1.0 reduces score, > 1.0 increases score)
    """
    # Base adjustment by grade (stricter = lower multiplier)
    grade_multipliers = {
        3: 1.15,  # Most lenient - boosts score
        4: 1.10,
        5: 1.05,  # Near middle
        6: 0.98,
        7: 0.92,
        8: 0.85,  # Strictest - reduces score most
    }
    
    multiplier = grade_multipliers.get(grade, 1.0)
    
    # Adjust based on essay length
    # Very short essays should be penalized more at higher grades
    if text_length < 50:
        multiplier *= 0.90  # Additional 10% penalty for very short essays at any grade
    elif text_length < 100:
        multiplier *= 0.95  # 5% penalty for short essays
    
    return multiplier



# Legacy component multipliers are kept as an optional fallback when
# dashboard metrics cannot be loaded at runtime.
LEGACY_COMPONENT_MULTIPLIERS = {
    3: {"richness": 1.0800, "organization": 1.0589, "technical": 1.0624},
    4: {"richness": 1.1760, "organization": 1.1232, "technical": 1.1994},
    5: {"richness": 1.0436, "organization": 1.0395, "technical": 1.0461},
    6: {"richness": 1.0215, "organization": 1.0194, "technical": 1.0206},
    7: {"richness": 1.0400, "organization": 1.0464, "technical": 1.0622},
    8: {"richness": 1.0735, "organization": 1.0619, "technical": 1.1020},
}


def apply_fairness_mitigation(score_dict: dict, dyslexic_flag: bool, grade: int) -> dict:
    # No adjustment for non-dyslexic students (baseline group).
    if not dyslexic_flag:
        score_dict["fairness_report"] = {
            "mitigation_applied": False,
            "reason": "Non-dyslexic student - no adjustment needed",
            "protected_attribute": "dyslexic_flag",
            "protected_value": False,
            "grade": grade,
        }
        return score_dict

    from .mitigation import mitigator

    metrics = mitigator.grade_metrics.get(grade)
    dynamic_active = mitigator.mitigation_active.get(grade, False)
    dynamic_multiplier = mitigator.calibration_multipliers.get(grade, 1.0)

    use_legacy_fallback = os.getenv("FAIRNESS_STATIC_FALLBACK", "0") == "1"

    if dynamic_active and dynamic_multiplier > 1.0:
        mult = {
            "richness": dynamic_multiplier,
            "organization": dynamic_multiplier,
            "technical": dynamic_multiplier,
        }
        method = "Conditional Proportional Mitigation (Dashboard-Driven)"
        data_source = "fairnessReports (Firebase)"
        justification = "Threshold-triggered mitigation from fairness dashboard metrics"
        fallback_used = False
    elif use_legacy_fallback:
        mult = LEGACY_COMPONENT_MULTIPLIERS.get(
            grade, {"richness": 1.0, "organization": 1.0, "technical": 1.0}
        )
        method = "Legacy Component Calibration (Fallback)"
        data_source = "component_bias_analysis.py empirical testing"
        justification = "Dashboard metrics unavailable - using configured legacy fallback"
        fallback_used = True
    else:
        mult = {"richness": 1.0, "organization": 1.0, "technical": 1.0}
        method = "No mitigation needed"
        data_source = "fairnessReports (Firebase)"
        if metrics is None:
            justification = "No fairness metrics loaded for this grade"
        else:
            justification = "No significant unfavorable bias found for this grade"
        fallback_used = False

    original_richness = score_dict.get("richness_5", 0.0)
    original_organization = score_dict.get("organization_6", 0.0)
    original_technical = score_dict.get("technical_3", 0.0)
    original_total = score_dict.get("total_14", 0.0)

    adjusted_richness = min(5.0, original_richness * mult["richness"])
    adjusted_organization = min(6.0, original_organization * mult["organization"])
    adjusted_technical = min(3.0, original_technical * mult["technical"])

    score_dict["richness_5"] = round(adjusted_richness, 2)
    score_dict["organization_6"] = round(adjusted_organization, 2)
    score_dict["technical_3"] = round(adjusted_technical, 2)
    score_dict["total_14"] = round(adjusted_richness + adjusted_organization + adjusted_technical, 2)

    mitigation_applied = (
        mult["richness"] > 1.01 or mult["organization"] > 1.01 or mult["technical"] > 1.01
    )

    score_dict["fairness_report"] = {
        "mitigation_applied": mitigation_applied,
        "method": method if mitigation_applied else "No mitigation needed",
        "protected_attribute": "dyslexic_flag",
        "protected_value": True,
        "grade": grade,
        "original_richness_5": round(original_richness, 2),
        "original_organization_6": round(original_organization, 2),
        "original_technical_3": round(original_technical, 2),
        "original_total_14": round(original_total, 2),
        "adjusted_richness_5": round(adjusted_richness, 2),
        "adjusted_organization_6": round(adjusted_organization, 2),
        "adjusted_technical_3": round(adjusted_technical, 2),
        "adjusted_total_14": score_dict["total_14"],
        "richness_multiplier": mult["richness"],
        "organization_multiplier": mult["organization"],
        "technical_multiplier": mult["technical"],
        "richness_boost": round(adjusted_richness - original_richness, 3),
        "organization_boost": round(adjusted_organization - original_organization, 3),
        "technical_boost": round(adjusted_technical - original_technical, 3),
        "total_boost": round(score_dict["total_14"] - original_total, 3),
        "justification": justification,
        "data_source": data_source,
        "fallback_used": fallback_used,
    }

    if metrics is not None:
        score_dict["fairness_report"]["dashboard_metrics"] = {
            "spd": round(metrics.spd, 3),
            "dir": round(metrics.dir, 3),
            "sample_size": metrics.sample_size,
            "thresholds_violated": bool(dynamic_active),
        }

    return score_dict



import re


def validate_sinhala_content(text: str) -> tuple[bool, str, str]:
    """
    Validate and CLEAN the text.
    - REJECTS if too little Sinhala content.
    - CLEANS mixed-script garbage (common in OCR) instead of rejecting entire text,
      unless the garbage overwhelms the text.
      
    Returns:
        (is_valid, reason_or_warning, cleaned_text)
    """
    if not text or not text.strip():
        return False, "Empty text", ""

    words = text.split()
    total_words = len(words)
    
    if total_words < 5:
        return True, "Short text", text

    sinhala_pattern = re.compile(r'[\u0D80-\u0DFF]')
    latin_digit_pattern = re.compile(r'[a-zA-Z0-9]')

    valid_sinhala_words = []
    mixed_garbage_count = 0
    pure_garbage_count = 0

    common_stops = {
        'සහ', 'හා', 'ද', 'ය', 'ගැන', 'විට', 'ලෙස', 'හෝ', 'නමුත්', 'නිසා', 
        'ලැබේ', 'කරයි', 'ඇත', 'නැත', 'මෙම', 'එම', 'අපි', 'ඔහු', 'ඇය', 
        'එය', 'මේ', 'ඒ', 'අර', 'විසින්', 'සමග', 'වැනි', 'බව'


    }
    
    x_pattern = re.compile(r'[^\u0D80-\u0DFFa-zA-Z0-9\s.,!?:;\"\'()\[\]{}]')

    for word in words:

        # 1. Strip external punctuation
        clean_word = word.strip(".,!?:;\"'()[]{}")

        
        has_sinhala = bool(sinhala_pattern.search(clean_word))
        has_latin_digit = bool(latin_digit_pattern.search(clean_word))
        has_symbols = bool(x_pattern.search(clean_word))

        is_mixed = has_sinhala and (has_latin_digit or has_symbols)
        
        if is_mixed:
            # STRIP mode: Remove only the non-Sinhala non-punctuation characters
            # Keep Sinhala chars and standard punctuation
            # This rescues suffixes like 'යෙන්' from 'DW76යෙන්'
            recovered_word = re.sub(r'[^\u0D80-\u0DFF]', '', clean_word)
            
            if len(recovered_word) > 0:
                valid_sinhala_words.append(recovered_word)
                mixed_garbage_count += 1
            else:
                pure_garbage_count += 1
            continue

        if has_latin_digit and not has_sinhala:
            # Pure latin/digit garbage
            pure_garbage_count += 1
            continue
            
        # Pure Sinhala (possibly with standard punctuation)
        valid_sinhala_words.append(word)


    cleaned_text = " ".join(valid_sinhala_words)
    cleaned_word_count = len(valid_sinhala_words)
    
    # Validation Logic
    if cleaned_word_count == 0:
        return False, "No valid Sinhala content found after cleaning", ""
        
    cleaning_ratio = cleaned_word_count / total_words
    
    # If we removed more than 40% of the text as garbage, it's probably bad input
    if cleaning_ratio < 0.6:
        return False, f"Too much garbage detected (kept {int(cleaning_ratio*100)}%)", ""

    return True, "Valid", cleaned_text



from .rubric_evaluator import rubric_evaluator

def score_sinhala_ml_v2(text: str, grade: int, dyslexic_flag: bool = False, topic: Optional[str] = None) -> dict:
    
    # 🔹 Validate and Clean Content
    is_valid, reason, cleaned_text = validate_sinhala_content(text)
    
    if not is_valid:
        print(f"[SINHALA-ML] Validation Failed: {reason}")
        return {
            "richness_5": 0.0,
            "organization_6": 0.0,
            "technical_3": 0.0,
            "total_14": 0.0,
            "fairness_report": {
                "mitigation_applied": False,
                "note": f"Scoring rejected due to invalid content: {reason}"
            }
        }
        
    # debug log
    if text != cleaned_text:
        print(f"[SINHALA-ML] OCR Cleaning applied. Removed garbage tokens.")

    # Get lazy-loaded model and tokenizer (will be None in test mode)
    model, tokenizer = load_model()
    
    # 🔹 CI-safe dummy output (no ML load)
    if model is None:
        # Apply grade adjustment even to dummy output
        text_length = len(cleaned_text.split())
        adjustment = _get_grade_adjustment_factor(grade, text_length)
        
        base_richness = 3.0
        base_organization = 3.5
        base_technical = 2.0
        base_total = 8.5
        
        scores = {
            "richness_5": round(base_richness * adjustment, 2),
            "organization_6": round(base_organization * adjustment, 2),
            "technical_3": round(base_technical * adjustment, 2),
            "total_14": round(base_total * adjustment, 2),
        }
        return apply_fairness_mitigation(scores, dyslexic_flag, grade)
    

    # Normalize punctuation before ML scoring to reduce punctuation-driven noise.
    model_input_text = rubric_evaluator.normalize_for_model(cleaned_text)
    if not model_input_text:
        model_input_text = cleaned_text

    enc = tokenizer(
        model_input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_ids = enc["input_ids"].to(DEVICE)
    attention_mask = enc["attention_mask"].to(DEVICE)

    # ✅ MUST be torch.long
    grade_tensor = torch.tensor([grade], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            grade_id=grade_tensor
        )
        
        # 🔹 Extract CLS for theme relevance if topic provided
        # Optimized: CLS is now returned by the model forward pass
        cls_emb = outputs.get("cls_emb")

    # Apply grade-aware adjustment to outputs
    words = cleaned_text.split()
    text_length = len(words)
    adjustment_factor = _get_grade_adjustment_factor(grade, text_length)
    
    # Base scores from ML model
    richness = float(outputs["richness_5"]) * adjustment_factor
    organization = float(outputs["organization_6"]) * adjustment_factor
    technical = float(outputs["technical_3"]) * adjustment_factor

    # ══════════════════════════════════════════════════════════════
    # HYBRID PHASE 1: Theme Relevance → affects richness_5
    # Uses XLM-R CLS cosine similarity (ML) + tiered penalty (Rule)
    # ══════════════════════════════════════════════════════════════
    relevance_score = 1.0
    # Safe logging
    if topic and cls_emb is not None:
        relevance_score = rubric_evaluator.compute_theme_relevance(
            cls_emb, topic, model, tokenizer, DEVICE,
            essay_text=cleaned_text
        )
        print(f"[HYBRID] Theme Relevance Score: {relevance_score:.3f}")

    # ══════════════════════════════════════════════════════════════
    # HYBRID PHASE 2: Technical Analysis → affects technical_3
    # Punctuation rules (6) + Heuristic Grammar (5 checks)
    # ══════════════════════════════════════════════════════════════
    tech_analysis = rubric_evaluator.analyze_technical(cleaned_text, grade=grade)
    technical_pre_rule = technical
    technical_after_penalty = technical_pre_rule - tech_analysis["penalty"]
    technical = min(technical_pre_rule, tech_analysis["technical_rule_score"])
    technical = max(0.3, technical)  # Floor of 0.3 for technical
    
    if tech_analysis["violations"]:
        print(f"[HYBRID] Punctuation Violations: {len(tech_analysis['violations'])}")
    if tech_analysis["grammar_issues"]:
        print(f"[HYBRID] Grammar Issues: {len(tech_analysis['grammar_issues'])}")

    # ══════════════════════════════════════════════════════════════
    # HYBRID PHASE 3: Richness Penalty → affects richness_5
    # Word Count (min 150) + Theme Relevance (cosine similarity)
    # ══════════════════════════════════════════════════════════════
    richness_penalty_info = rubric_evaluator.compute_richness_penalty(
        relevance_score, text_length
    )
    
    # Apply theme penalty
    if richness_penalty_info["theme_penalty"] > 0:
        richness -= richness_penalty_info["theme_penalty"]
        print(f"[HYBRID] Theme Penalty: -{richness_penalty_info['theme_penalty']}")
    
    # Apply word count penalty
    if richness_penalty_info["word_count_penalty"] > 0:
        richness -= richness_penalty_info["word_count_penalty"]
        print(f"[HYBRID] Word Count Penalty: -{richness_penalty_info['word_count_penalty']} "
              f"(Length: {text_length}/150)")
    
    # Floor for richness, except for completely off-topic essays.
    if richness_penalty_info.get("force_richness_zero", False):
        richness = 0.0
    else:
        richness = max(0.3, richness)

    # ══════════════════════════════════════════════════════════════
    # FINAL SCORING
    # ══════════════════════════════════════════════════════════════
    # Clamp individual scores to their rubric maximums
    richness = min(5.0, richness)
    organization = min(6.0, organization)
    technical = min(3.0, technical)
    
    scores = {
        "richness_5": round(richness, 2),
        "organization_6": round(organization, 2),
        "technical_3": round(technical, 2),
        "total_14": round(min(14.0, richness + organization + technical), 2),
    }

    result = apply_fairness_mitigation(scores, dyslexic_flag, grade)
    
    # ══════════════════════════════════════════════════════════════
    # TRANSPARENCY: Full Rubric Report for Research Logging
    # ══════════════════════════════════════════════════════════════
    if "fairness_report" not in result:
        result["fairness_report"] = {}
    
    result["fairness_report"]["rubric_notes"] = {
        "scoring_method": "Hybrid (ML + Rule-Based)",
        "theme_relevance": round(relevance_score, 3),
        "theme_penalty": richness_penalty_info["theme_penalty"],
        "word_count": text_length,
        "word_count_penalty": richness_penalty_info["word_count_penalty"],
        "force_richness_zero": bool(richness_penalty_info.get("force_richness_zero", False)),
        "model_input_word_count": len(model_input_text.split()),
        "technical_violations": tech_analysis["violations"],
        "grammar_issues": tech_analysis["grammar_issues"],
        "technical_penalty": tech_analysis["penalty"],
        "technical_punctuation_penalty": tech_analysis.get("punctuation_penalty", 0.0),
        "technical_grammar_penalty": tech_analysis.get("grammar_penalty", 0.0),
        "grammar_checks_evaluated": bool(tech_analysis.get("grammar_checks_evaluated", True)),
        "technical_rule_hits": tech_analysis.get("rule_hits", {}),
        "technical_breakdown": tech_analysis.get("technical_breakdown", {}),
        "technical_rule_cap": tech_analysis.get("rule_based_technical_cap", 3.0),
        "technical_pre_rule_score": round(technical_pre_rule, 3),
        "technical_after_penalty_pre_cap": round(technical_after_penalty, 3),
        "model_punctuation_normalized": True,
        "grade_adjustment_factor": round(adjustment_factor, 3)
    }

    # Add note about cleaning if applicable
    if text != cleaned_text:
        existing_note = result["fairness_report"].get("note", "")
        result["fairness_report"]["note"] = (existing_note + " [OCR Garbage Cleaned]").strip()
        
    return result


