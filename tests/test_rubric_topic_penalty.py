import torch

import app.sinhala_ml_v2 as sinhala_ml_v2
from app.rubric_evaluator import rubric_evaluator


def test_theme_penalty_macro_tiers_match_spec():
    # (>0.8 -> 0), (0.7-0.8 -> 0.5), (0.6-0.7 -> 1), (0.5-0.6 -> 2),
    # (0.4-0.5 -> 3), (<0.4 -> 4..5)
    cases = [
        (0.81, 0.0),
        (0.80, 0.5),
        (0.70, 0.5),
        (0.69, 1.0),
        (0.60, 1.0),
        (0.59, 2.0),
        (0.50, 2.0),
        (0.49, 3.0),
        (0.40, 3.0),
        (0.39, 4.0),
        (0.20, 4.0),
        (0.19, 5.0),
    ]

    for relevance, expected_penalty in cases:
        penalty = rubric_evaluator.compute_richness_penalty(relevance, word_count=150)
        assert penalty["theme_penalty"] == expected_penalty


def test_force_richness_zero_threshold_is_below_point_two_five():
    assert rubric_evaluator.compute_richness_penalty(0.24, word_count=150)["force_richness_zero"] is True
    assert rubric_evaluator.compute_richness_penalty(0.25, word_count=150)["force_richness_zero"] is False


def test_word_count_penalty_tiers_match_spec():
    cases = [
        (150, 0.0),
        (149, 0.5),
        (130, 0.5),
        (129, 1.0),
        (110, 1.0),
        (109, 2.0),
        (90, 2.0),
        (89, 3.0),
        (24, 3.0),
    ]

    for word_count, expected_penalty in cases:
        penalty = rubric_evaluator.compute_richness_penalty(0.81, word_count=word_count)
        assert penalty["word_count_penalty"] == expected_penalty


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
            "technical_3": 2.0,
            "cls_emb": torch.ones((1, 4), dtype=torch.float32),
        }


def test_completely_off_topic_forces_richness_zero(monkeypatch):
    monkeypatch.setattr(sinhala_ml_v2, "load_model", lambda: (_FakeModel(), _FakeTokenizer()))
    monkeypatch.setattr(rubric_evaluator, "compute_theme_relevance", lambda *args, **kwargs: 0.05)

    text = " ".join(["වැස්ස"] * 160)
    result = sinhala_ml_v2.score_sinhala_ml_v2(
        text,
        grade=6,
        dyslexic_flag=False,
        topic="මගේ පාසල",
    )

    assert result["richness_5"] == 0.0
    assert result["fairness_report"]["rubric_notes"]["theme_penalty"] == 5.0
    assert result["fairness_report"]["rubric_notes"]["force_richness_zero"] is True
