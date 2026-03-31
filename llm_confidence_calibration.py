import re
from enum import Enum
from typing import Dict, Tuple


class SignalStrength(Enum):
    CLEAR = 0.8
    MODERATE = 0.5
    VAGUE = 0.45
    INSUFFICIENT = 0.2


class ConfidenceCalibrator:
    CLEAR_SYMPTOMS = {
        "acidity",
        "burning sensation",
        "high blood pressure",
        "hypertension",
        "diabetes",
        "thyroid",
        "allergy",
        "intolerance",
        "kidney",
        "liver",
        "heart",
    }

    VAGUE_TERMS = {
        "sometimes",
        "occasionally",
        "maybe",
        "i feel off",
        "feel off",
        "not sure",
        "randomly",
    }

    @classmethod
    def analyze_input(cls, text: str) -> SignalStrength:
        score = cls.get_confidence_score(text)
        if score > 0.7:
            return SignalStrength.CLEAR
        if score >= 0.4:
            return SignalStrength.MODERATE
        if score >= 0.25:
            return SignalStrength.VAGUE
        return SignalStrength.INSUFFICIENT

    @classmethod
    def get_confidence_score(cls, text: str) -> float:
        if not text or not text.strip():
            return 0.0

        sample = text.lower().strip()
        tokens = re.findall(r"[a-z]+", sample)
        if not tokens:
            return 0.1

        clear_hits = sum(1 for item in cls.CLEAR_SYMPTOMS if item in sample)
        vague_hits = sum(1 for item in cls.VAGUE_TERMS if item in sample)

        # Detect obvious gibberish like "asdfgh"
        if clear_hits == 0 and vague_hits == 0 and len(tokens) <= 3 and all(len(t) >= 5 for t in tokens):
            return 0.15

        if clear_hits >= 2:
            return 0.82
        if clear_hits == 1:
            return 0.74
        if vague_hits > 0:
            return 0.5

        return 0.3

    @classmethod
    def detect_ambiguity(cls, text: str) -> Dict[str, object]:
        sample = (text or "").lower()
        vague_terms_found = [item for item in cls.VAGUE_TERMS if item in sample]
        too_short = len(sample.split()) < 4

        reduction = 0.0
        if vague_terms_found:
            reduction += 0.1
        if too_short:
            reduction += 0.1

        return {
            "has_vague_terms": bool(vague_terms_found),
            "has_multiple_interpretations": bool(vague_terms_found),
            "lacks_specifics": not bool(vague_terms_found),
            "too_short": too_short,
            "vague_terms_found": vague_terms_found,
            "confidence_reduction": reduction,
        }

    @classmethod
    def calibrate_confidence(cls, base_confidence: float, ambiguity_analysis: Dict[str, object]) -> float:
        reduction = float(ambiguity_analysis.get("confidence_reduction", 0.0))
        calibrated = max(0.0, min(1.0, float(base_confidence) - reduction))
        return round(calibrated, 2)


class DoshaConfidenceCalibrator:
    @classmethod
    def assess_dosha_confidence(
        cls,
        has_age: bool,
        has_weight: bool,
        has_height: bool,
        has_symptoms: bool,
        symptom_clarity: float,
        data_consistency: bool,
    ) -> float:
        score = 0.0
        if has_age:
            score += 0.15
        if has_weight:
            score += 0.15
        if has_height:
            score += 0.1
        if has_symptoms:
            score += 0.4 * max(0.0, min(1.0, symptom_clarity))
        if data_consistency:
            score += 0.2

        if not has_symptoms:
            score = min(score, 0.4)
        return round(min(score, 1.0), 2)

    @classmethod
    def detect_dosha_indicators(cls, text: str) -> Tuple[Dict[str, int], float]:
        sample = (text or "").lower()
        indicators = {
            "vata": int("dry" in sample) + int("cold" in sample),
            "pitta": int("hot" in sample) + int("burning" in sample),
            "kapha": int("heavy" in sample) + int("sluggish" in sample),
        }
        total = sum(indicators.values())
        return indicators, round(min(total / 4, 1.0), 2)


class ConfidenceThresholds:
    HIGH_CONFIDENCE = 0.7
    MEDIUM_CONFIDENCE = 0.45
    LOW_CONFIDENCE = 0.25

    @classmethod
    def should_process(cls, confidence: float) -> bool:
        return confidence >= cls.MEDIUM_CONFIDENCE

    @classmethod
    def should_use_fallback(cls, confidence: float) -> bool:
        return confidence < cls.LOW_CONFIDENCE

    @classmethod
    def describe_confidence(cls, confidence: float) -> str:
        if confidence >= cls.HIGH_CONFIDENCE:
            return "HIGH"
        if confidence >= cls.MEDIUM_CONFIDENCE:
            return "MEDIUM"
        if confidence >= cls.LOW_CONFIDENCE:
            return "LOW"
        return "VERY_LOW"
