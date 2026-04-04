import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from llm_schemas_strict import (
    AIProfileOutput_v1,
    FeedbackType,
    RiskFlag,
    StructuredExplanation_v1,
    StructuredFeedback_v1,
)


class HallucinationDetector:
    VALID_RISK_FLAGS = {flag.value for flag in RiskFlag}

    @staticmethod
    def is_valid_risk_flag(flag: str) -> bool:
        return isinstance(flag, str) and flag in HallucinationDetector.VALID_RISK_FLAGS

    @staticmethod
    def has_ungrounded_claims(explanation: str) -> bool:
        patterns = (
            r"\bproven to cure\b",
            r"\bguaranteed\b",
            r"\balways works\b",
            r"\b100% effective\b",
            r"\bmiracle\b",
        )
        text = explanation.lower()
        return any(re.search(pattern, text) for pattern in patterns)

    @staticmethod
    def has_fabricated_citations(sources: List[str]) -> bool:
        if not isinstance(sources, list):
            return True
        for source in sources:
            if not isinstance(source, str):
                return True
            if not source.strip():
                return True
        return False


class OutputValidator:
    PROFILE_FIELDS = {"risk_flags", "dosha_estimate", "confidence"}
    FEEDBACK_FIELDS = {"feedback_type", "target"}
    EXPLANATION_FIELDS = {"explanation", "sources"}

    @classmethod
    def validate_profile_output(cls, raw_data: Dict) -> Tuple[bool, Optional[AIProfileOutput_v1], str]:
        try:
            if set(raw_data.keys()) != cls.PROFILE_FIELDS:
                return False, None, "profile output must contain exactly risk_flags, dosha_estimate, confidence"

            flags = raw_data.get("risk_flags", [])
            if not isinstance(flags, list):
                return False, None, "risk_flags must be a list"
            if not flags:
                flags = [RiskFlag.NONE.value]
            for flag in flags:
                if not HallucinationDetector.is_valid_risk_flag(flag):
                    return False, None, f"invalid risk flag: {flag}"

            output = AIProfileOutput_v1.model_validate(
                {
                    "risk_flags": flags,
                    "dosha_estimate": raw_data.get("dosha_estimate", {}),
                    "confidence": raw_data.get("confidence", 0.2),
                }
            )
            return True, output, ""
        except Exception as exc:
            return False, None, f"validation error: {exc}"

    @classmethod
    def validate_feedback_output(cls, raw_data: Dict) -> Tuple[bool, Optional[StructuredFeedback_v1], str]:
        try:
            if set(raw_data.keys()) != cls.FEEDBACK_FIELDS:
                return False, None, "feedback output must contain exactly feedback_type, target"

            feedback_type = str(raw_data.get("feedback_type", "")).upper()
            if feedback_type not in {t.value for t in FeedbackType}:
                feedback_type = FeedbackType.DISLIKE.value

            output = StructuredFeedback_v1.model_validate(
                {"feedback_type": feedback_type, "target": str(raw_data.get("target", ""))}
            )
            return True, output, ""
        except Exception as exc:
            return False, None, f"validation error: {exc}"

    @classmethod
    def validate_explanation_output(
        cls,
        raw_data: Dict,
    ) -> Tuple[bool, Optional[StructuredExplanation_v1], str]:
        try:
            if set(raw_data.keys()) != cls.EXPLANATION_FIELDS:
                return False, None, "explanation output must contain exactly explanation, sources"

            explanation = str(raw_data.get("explanation", ""))
            sources = raw_data.get("sources", [])

            if HallucinationDetector.has_ungrounded_claims(explanation):
                return False, None, "ungrounded medical claim detected"
            if HallucinationDetector.has_fabricated_citations(sources):
                return False, None, "invalid sources"

            output = StructuredExplanation_v1.model_validate(
                {"explanation": explanation, "sources": sources}
            )
            return True, output, ""
        except Exception as exc:
            return False, None, f"validation error: {exc}"


class ContextValidator:
    @staticmethod
    def _normalize(text: str) -> str:
        cleaned = re.sub(r"\s+", " ", (text or "").strip().lower())
        return re.sub(r"[^a-z0-9\s\.\,\-\(\)]", "", cleaned)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        sample = ContextValidator._normalize(text)
        parts = re.split(r"[.!?]+", sample)
        return [p.strip() for p in parts if len(p.strip()) >= 12]

    @staticmethod
    def _sentence_supported(sentence: str, context_sentences: List[str]) -> bool:
        if not sentence:
            return True
        for chunk in context_sentences:
            if sentence in chunk or chunk in sentence:
                return True
            ratio = SequenceMatcher(None, sentence, chunk).ratio()
            if ratio >= 0.82:
                return True
            sentence_tokens = set(re.findall(r"\b[a-z0-9]{4,}\b", sentence))
            chunk_tokens = set(re.findall(r"\b[a-z0-9]{4,}\b", chunk))
            if sentence_tokens:
                overlap = len(sentence_tokens & chunk_tokens) / len(sentence_tokens)
                if overlap >= 0.75:
                    return True
        return False

    @staticmethod
    def check_context_adherence(explanation: str, context_chunks: List[str]) -> Tuple[bool, float]:
        if not context_chunks:
            return True, 0.0

        explanation_sentences = ContextValidator._split_sentences(explanation)
        context_sentences: List[str] = []
        for chunk in context_chunks:
            context_sentences.extend(ContextValidator._split_sentences(chunk))

        if not explanation_sentences or not context_sentences:
            return False, 1.0

        supported = sum(
            1 for sentence in explanation_sentences if ContextValidator._sentence_supported(sentence, context_sentences)
        )
        adherence = supported / len(explanation_sentences)
        has_external_claims = adherence < 0.7
        return has_external_claims, adherence

    @staticmethod
    def confidence_penalty(adherence: float) -> float:
        score = max(0.0, min(1.0, float(adherence)))
        if score >= 0.9:
            return 0.0
        if score >= 0.75:
            return 0.1
        if score >= 0.6:
            return 0.2
        return 0.35


class FallbackGenerator:
    @staticmethod
    def create_fallback_profile(confidence: float = 0.2) -> Dict:
        safe_confidence = max(0.0, min(float(confidence), 1.0))
        return {
            "risk_flags": [RiskFlag.NONE.value],
            "dosha_estimate": {"vata": 0.33, "pitta": 0.33, "kapha": 0.34},
            "confidence": round(safe_confidence, 2),
        }

    @staticmethod
    def create_fallback_feedback() -> Dict:
        return {"feedback_type": FeedbackType.DISLIKE.value, "target": ""}

    @staticmethod
    def create_fallback_explanation() -> Dict:
        return {"explanation": "", "sources": []}
