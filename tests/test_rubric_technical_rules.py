import torch

from app.rubric_evaluator import rubric_evaluator
import app.sinhala_ml_v2 as sinhala_ml_v2


def _build_sinhala_sentences(count: int, missing_endings: set[int] | None = None) -> str:
    missing_endings = missing_endings or set()
    lines = []
    for i in range(count):
        sentence = "මම පාසලට ගොස් පාඩම් කරමි"
        if i not in missing_endings:
            sentence += "."
        lines.append(sentence)
    # Keep each logical sentence on its own line so missing end punctuation
    # can be measured reliably by the rubric sentence-end checker.
    return "\n".join(lines)


def test_technical_breakdown_allocation_totals_to_three():
    text = _build_sinhala_sentences(12)
    analysis = rubric_evaluator.analyze_technical(text)
    breakdown = analysis["technical_breakdown"]

    total = (
        breakdown["punctuation_score_1"]
        + breakdown["word_separation_score_0_5"]
        + breakdown["paragraph_structure_score_0_75"]
        + breakdown["layout_structure_score_0_75"]
    )
    assert abs(total - analysis["technical_rule_score"]) < 1e-6
    assert 0.0 <= analysis["technical_rule_score"] <= 3.0


def test_supported_sinhala_punctuation_symbols_exposed():
    text = _build_sinhala_sentences(3)
    analysis = rubric_evaluator.analyze_technical(text)
    supported = analysis["technical_breakdown"]["punctuation_symbols_supported"]
    expected = [".", ",", ";", ":", "?", "!", "“ ”", "‘ ’", "-", "–", "…", "()"]
    assert supported == expected


def test_punctuation_tier_deduction_five_percent():
    # 1 missing out of 20 sentences -> 5% error -> deduction 0.1 -> score 0.9
    text = _build_sinhala_sentences(20, missing_endings={19})
    analysis = rubric_evaluator.analyze_technical(text)
    assert analysis["technical_breakdown"]["sentence_count"] == 20
    assert abs(analysis["technical_breakdown"]["punctuation_error_rate"] - 0.05) < 1e-6
    assert analysis["technical_breakdown"]["punctuation_score_1"] == 0.9


def test_punctuation_tier_deduction_ten_to_twenty_percent():
    # 2 missing out of 10 sentences -> 20% error -> deduction 0.4 -> score 0.6
    text = _build_sinhala_sentences(10, missing_endings={2, 7})
    analysis = rubric_evaluator.analyze_technical(text)
    assert abs(analysis["technical_breakdown"]["punctuation_error_rate"] - 0.2) < 1e-6
    assert analysis["technical_breakdown"]["punctuation_score_1"] == 0.6


def test_punctuation_tier_deduction_above_forty_percent():
    # 5 missing out of 10 sentences -> 50% error -> full punctuation loss
    text = _build_sinhala_sentences(10, missing_endings={0, 1, 2, 3, 4})
    analysis = rubric_evaluator.analyze_technical(text)
    assert analysis["technical_breakdown"]["punctuation_error_rate"] == 0.5
    assert analysis["technical_breakdown"]["punctuation_score_1"] == 0.0
    assert analysis["punctuation_penalty"] == 1.0


def test_long_sentence_no_punctuation_structural_penalty():
    long_words = " ".join(["වචනය"] * 36)
    without_punct = long_words
    with_punct = long_words + "."

    no_punct = rubric_evaluator.analyze_technical(without_punct)
    with_end = rubric_evaluator.analyze_technical(with_punct)

    no_punct_par = no_punct["technical_breakdown"]["paragraph_structure_score_0_75"]
    with_end_par = with_end["technical_breakdown"]["paragraph_structure_score_0_75"]

    assert no_punct["technical_breakdown"]["long_sentence_no_punctuation_count"] == 1
    assert with_end["technical_breakdown"]["long_sentence_no_punctuation_count"] == 0
    assert round(with_end_par - no_punct_par, 2) >= 0.2


class _FakeTokenizer:
    def __call__(self, text, return_tensors="pt", truncation=True, max_length=512, padding=False):
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
        }


class _FakeModel:
    def __call__(self, input_ids, attention_mask, grade_id):
        return {
            "richness_5": 4.0,
            "organization_6": 4.5,
            "technical_3": 2.7,
            "cls_emb": None,
        }


def test_punctuation_changes_only_technical_not_richness_or_organization(monkeypatch):
    monkeypatch.setattr(sinhala_ml_v2, "load_model", lambda: (_FakeModel(), _FakeTokenizer()))

    base = _build_sinhala_sentences(40)
    changed = _build_sinhala_sentences(40, missing_endings={39})

    a = sinhala_ml_v2.score_sinhala_ml_v2(base, grade=6, dyslexic_flag=False, topic=None)
    b = sinhala_ml_v2.score_sinhala_ml_v2(changed, grade=6, dyslexic_flag=False, topic=None)

    assert a["richness_5"] == b["richness_5"]
    assert a["organization_6"] == b["organization_6"]
    assert b["technical_3"] <= a["technical_3"]


def test_one_missing_full_stop_has_small_effect_with_many_sentences():
    with_all = _build_sinhala_sentences(40)
    missing_one = _build_sinhala_sentences(40, missing_endings={0})

    a = rubric_evaluator.analyze_technical(with_all)
    b = rubric_evaluator.analyze_technical(missing_one)

    # 1/40 = 2.5% -> only 0.1 punctuation deduction
    assert b["technical_breakdown"]["punctuation_error_rate"] == 0.025
    assert round(
        a["technical_breakdown"]["punctuation_score_1"]
        - b["technical_breakdown"]["punctuation_score_1"],
        2,
    ) == 0.1


def test_low_grades_do_not_emit_grammar_issue_flags():
    text = "වචන ටිකක් මෙහි ලියනවා"
    low_grade = rubric_evaluator.analyze_technical(text, grade=3)
    upper_grade = rubric_evaluator.analyze_technical(text, grade=7)

    assert low_grade["grammar_checks_evaluated"] is False
    assert low_grade["grammar_issues"] == []
    assert upper_grade["grammar_checks_evaluated"] is True
