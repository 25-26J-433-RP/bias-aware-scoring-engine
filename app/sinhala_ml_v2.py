# app/sinhala_ml_v2.py

import os
import torch
from typing import Optional

MODEL_SOURCE = "akura-official/xlm-roberta-large-sinhala-multihead"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 🔹 Detect CI / test environment
IS_TEST = os.getenv("DISABLE_ML", "0") == "1"

# Lazy-loaded model and tokenizer (initialized on first use)
_tokenizer = None
_model = None


def load_model():
    """
    Lazy load both model and tokenizer on first use.
    This ensures NO loading happens at import time.
    """
    global _model, _tokenizer
    
    if _model is not None:
        return _model, _tokenizer
    
    if IS_TEST:
        return None, None
    
    print("[SINHALA-ML] Loading model and tokenizer...")
    
    # Import inside function to avoid any import-time side effects
    from transformers import AutoTokenizer
    from .model_multitask_xlmr import SinhalaMultiHeadRegressor
    
    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            MODEL_SOURCE,
            use_fast=False,
            trust_remote_code=True
        )
        print("[SINHALA-ML] Tokenizer loaded.")
        
        _model = SinhalaMultiHeadRegressor.from_pretrained(
            MODEL_SOURCE,
            trust_remote_code=True
        )
        _model.to(DEVICE)
        _model.eval()
        print("[SINHALA-ML] Model loaded successfully.")
        
    except Exception as e:
        print(f"[SINHALA-ML] Error loading model: {e}")
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



# Component-specific calibration based on empirical bias analysis
# Source: component_bias_analysis.py statistical testing
# Multipliers derived from actual mean score gaps between groups

def apply_fairness_mitigation(score_dict: dict, dyslexic_flag: bool, grade: int) -> dict:
    print(f"[FAIRNESS] Mitigation check: Dyslexic={dyslexic_flag}, Grade={grade}")
    
    # No adjustment for non-dyslexic students (they are the fairness baseline)
    if not dyslexic_flag:
        score_dict['fairness_report'] = {
            "mitigation_applied": False,
            "reason": "Non-dyslexic student - no adjustment needed",
            "protected_attribute": "dyslexic_flag",
            "protected_value": False,
            "grade": grade,
        }
        return score_dict
    
    # Component-specific multipliers from Firestore data analysis
    # Formula: multiplier = mean_non_dyslexic / mean_dyslexic
    # Source: analysis/calculate_exact_multipliers.py (run on 2026-02-18)
    # These are EXACT values, not estimates
    CALIBRATION_MULTIPLIERS = {
        3: {
            "richness": 1.0800,
            "organization": 1.0589,
            "technical": 1.0624,
        },  # n_dys=35, n_non_dys=33
        4: {
            "richness": 1.1760,
            "organization": 1.1232,
            "technical": 1.1994,
        },  # n_dys=25, n_non_dys=13 (SEVERE bias)
        5: {
            "richness": 1.0436,
            "organization": 1.0395,
            "technical": 1.0461,
        },  # n_dys=24, n_non_dys=18
        6: {
            "richness": 1.0215,
            "organization": 1.0194,
            "technical": 1.0206,
        },  # n_dys=27, n_non_dys=53 (small effect, apply minimal adjustment)
        7: {
            "richness": 1.0400,
            "organization": 1.0464,
            "technical": 1.0622,
        },  # n_dys=37, n_non_dys=41
        8: {
            "richness": 1.0735,
            "organization": 1.0619,
            "technical": 1.1020,
        },  # n_dys=79, n_non_dys=91 (small but statistically significant)
    }
    
    mult = CALIBRATION_MULTIPLIERS.get(grade, {"richness": 1.0, "organization": 1.0, "technical": 1.0})
    
    # CRITICAL: Only apply if student is dyslexic
    if not dyslexic_flag:
        mult = {"richness": 1.0, "organization": 1.0, "technical": 1.0}
    
    # Store original values for transparency
    original_richness = score_dict.get("richness_5", 0)
    original_organization = score_dict.get("organization_6", 0)
    original_technical = score_dict.get("technical_3", 0)
    original_total = score_dict.get("total_14", 0)
    
    # Apply component-specific calibration
    adjusted_richness = min(5.0, original_richness * mult["richness"])
    adjusted_organization = min(6.0, original_organization * mult["organization"])
    adjusted_technical = min(3.0, original_technical * mult["technical"])
    
    # Update scores
    score_dict["richness_5"] = round(adjusted_richness, 2)
    score_dict["organization_6"] = round(adjusted_organization, 2)
    score_dict["technical_3"] = round(adjusted_technical, 2)
    score_dict["total_14"] = round(adjusted_richness + adjusted_organization + adjusted_technical, 2)
    
    # Determine if meaningful mitigation was applied
    mitigation_applied = (mult["richness"] > 1.01 or mult["organization"] > 1.01 or mult["technical"] > 1.01)
    
    # Full transparency reporting
    score_dict["fairness_report"] = {
        "mitigation_applied": mitigation_applied,
        "method": "Component-Specific Calibration (Empirical)" if mitigation_applied else "No mitigation needed",
        "protected_attribute": "dyslexic_flag",
        "protected_value": True,
        "grade": grade,
        # Original values
        "original_richness_5": round(original_richness, 2),
        "original_organization_6": round(original_organization, 2),
        "original_technical_3": round(original_technical, 2),
        "original_total_14": round(original_total, 2),
        # Adjusted values (same as in score_dict)
        "adjusted_richness_5": round(adjusted_richness, 2),
        "adjusted_organization_6": round(adjusted_organization, 2),
        "adjusted_technical_3": round(adjusted_technical, 2),
        "adjusted_total_14": score_dict["total_14"],
        # Multipliers applied
        "richness_multiplier": mult["richness"],
        "organization_multiplier": mult["organization"],
        "technical_multiplier": mult["technical"],
        # Absolute boost
        "richness_boost": round(adjusted_richness - original_richness, 3),
        "organization_boost": round(adjusted_organization - original_organization, 3),
        "technical_boost": round(adjusted_technical - original_technical, 3),
        "total_boost": round(score_dict["total_14"] - original_total, 3),
        # Scientific justification
        "justification": "Statistical analysis showed significant ML model bias for this grade" if mitigation_applied 
                         else ("Non-dyslexic student - no adjustment needed" if not dyslexic_flag 
                               else "No significant bias found for this grade"),
        "data_source": "component_bias_analysis.py empirical testing",
    }
    
    print(f"[FAIRNESS] Applied? {mitigation_applied} | Boost: {score_dict['fairness_report'].get('total_boost', 0)}")
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
    

    enc = tokenizer(
        cleaned_text,
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
        cls_emb = None
        if topic and topic.strip():
            # We can re-run encoder but more efficient to just use existing model output if accessible
            # However, SinhalaMultiHeadRegressor doesn't return CLS. Let's get it manually.
            out_encoder = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = out_encoder.last_hidden_state[:, 0, :]

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
    print(f"[HYBRID] Topic received: '{topic}' (type={type(topic).__name__})")
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
    tech_analysis = rubric_evaluator.analyze_technical(cleaned_text)
    technical -= tech_analysis["penalty"]
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
    
    # Floor for richness
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
        "technical_violations": tech_analysis["violations"],
        "grammar_issues": tech_analysis["grammar_issues"],
        "technical_penalty": tech_analysis["penalty"],
        "grade_adjustment_factor": round(adjustment_factor, 3)
    }

    # Add note about cleaning if applicable
    if text != cleaned_text:
        existing_note = result["fairness_report"].get("note", "")
        result["fairness_report"]["note"] = (existing_note + " [OCR Garbage Cleaned]").strip()
        
    return result


