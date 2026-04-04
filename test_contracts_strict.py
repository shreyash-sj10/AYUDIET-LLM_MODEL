from llm_hallucination_control import OutputValidator


def test_profile_contract_accepts_valid_schema():
    payload = {
        "risk_flags": ["diabetes"],
        "dosha_estimate": {"vata": 0.2, "pitta": 0.5, "kapha": 0.3},
        "confidence": 0.85,
    }
    ok, parsed, err = OutputValidator.validate_profile_output(payload)
    assert ok, err
    assert parsed is not None


def test_profile_contract_rejects_bad_dosha_sum():
    payload = {
        "risk_flags": ["diabetes"],
        "dosha_estimate": {"vata": 0.2, "pitta": 0.2, "kapha": 0.2},
        "confidence": 0.85,
    }
    ok, _, _ = OutputValidator.validate_profile_output(payload)
    assert not ok


def test_profile_contract_rejects_unknown_risk_flag():
    payload = {
        "risk_flags": ["made_up_condition"],
        "dosha_estimate": {"vata": 0.3, "pitta": 0.3, "kapha": 0.4},
        "confidence": 0.5,
    }
    ok, _, _ = OutputValidator.validate_profile_output(payload)
    assert not ok


def test_feedback_contract_defaults_invalid_type_to_dislike():
    payload = {"feedback_type": "MAYBE", "target": "item"}
    ok, parsed, err = OutputValidator.validate_feedback_output(payload)
    assert ok, err
    assert parsed is not None
    assert parsed.feedback_type.value == "DISLIKE"
