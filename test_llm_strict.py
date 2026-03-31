import json

from llm_confidence_calibration import ConfidenceCalibrator
from llm_hallucination_control import FallbackGenerator, OutputValidator
from llm_schemas_strict import FeedbackType, RiskFlag


def test_case_clear_input_high_confidence():
    user_input = "I have acidity and burning sensation"
    confidence = ConfidenceCalibrator.get_confidence_score(user_input)
    assert confidence > 0.7


def test_case_vague_input_medium_low_confidence():
    user_input = "I feel off sometimes"
    confidence = ConfidenceCalibrator.get_confidence_score(user_input)
    assert 0.4 <= confidence <= 0.6


def test_case_invalid_input_fallback_output():
    user_input = "asdfgh"
    confidence = ConfidenceCalibrator.get_confidence_score(user_input)
    assert confidence < 0.4

    fallback = FallbackGenerator.create_fallback_profile(confidence)
    assert fallback == {
        "risk_flags": [RiskFlag.NONE.value],
        "dosha_estimate": {"vata": 0.33, "pitta": 0.33, "kapha": 0.34},
        "confidence": round(confidence, 2),
    }


def test_profile_schema_strict_fields_and_sum():
    payload = {
        "risk_flags": ["diabetes"],
        "dosha_estimate": {"vata": 0.2, "pitta": 0.5, "kapha": 0.3},
        "confidence": 0.8,
    }
    is_valid, parsed, error = OutputValidator.validate_profile_output(payload)
    assert is_valid, error
    assert parsed is not None


def test_profile_rejects_extra_fields():
    payload = {
        "risk_flags": ["none"],
        "dosha_estimate": {"vata": 0.33, "pitta": 0.33, "kapha": 0.34},
        "confidence": 0.2,
        "extra": "not allowed",
    }
    is_valid, _, _ = OutputValidator.validate_profile_output(payload)
    assert not is_valid


def test_feedback_unclear_defaults_to_dislike():
    payload = {"feedback_type": "MAYBE", "target": "recipe 1"}
    is_valid, parsed, error = OutputValidator.validate_feedback_output(payload)
    assert is_valid, error
    assert parsed is not None
    assert parsed.feedback_type == FeedbackType.DISLIKE


def test_explanation_schema_strict():
    payload = {
        "explanation": "Use only retrieved context. (Source: Charaka Samhita, Chapter 1)",
        "sources": ["Charaka Samhita, Chapter 1"],
    }
    is_valid, parsed, error = OutputValidator.validate_explanation_output(payload)
    assert is_valid, error
    assert parsed is not None


def test_invalid_json_example_uses_fallback():
    bad_json = "{'x':1"
    try:
        json.loads(bad_json)
        assert False, "expected JSONDecodeError"
    except json.JSONDecodeError:
        assert FallbackGenerator.create_fallback_feedback() == {
            "feedback_type": FeedbackType.DISLIKE.value,
            "target": "",
        }
