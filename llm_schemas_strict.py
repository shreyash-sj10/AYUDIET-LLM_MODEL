from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class RiskFlag(str, Enum):
    HIGH_BLOOD_PRESSURE = "high_blood_pressure"
    DIABETES = "diabetes"
    KIDNEY_DISEASE = "kidney_disease"
    LIVER_DISEASE = "liver_disease"
    HEART_CONDITION = "heart_condition"
    THYROID_DISORDER = "thyroid_disorder"
    ALLERGY = "allergy"
    INTOLERANCE = "intolerance"
    NONE = "none"


class FeedbackType(str, Enum):
    LIKE = "LIKE"
    DISLIKE = "DISLIKE"
    REPLACE = "REPLACE"


class DoshaScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vata: float = Field(ge=0.0, le=1.0)
    pitta: float = Field(ge=0.0, le=1.0)
    kapha: float = Field(ge=0.0, le=1.0)

    @field_validator("vata", "pitta", "kapha", mode="before")
    @classmethod
    def _round_score(cls, value: Any) -> float:
        return round(float(value), 2)

    @model_validator(mode="after")
    def _validate_sum(self) -> "DoshaScores":
        total = self.vata + self.pitta + self.kapha
        if abs(total - 1.0) > 0.02:
            raise ValueError("dosha_estimate must sum to ~1.0")
        return self


class AIProfileOutput_v1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_flags: List[RiskFlag]
    dosha_estimate: DoshaScores
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("confidence", mode="before")
    @classmethod
    def _round_confidence(cls, value: Any) -> float:
        return round(float(value), 2)


class StructuredExplanation_v1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanation: str = Field(max_length=2000)
    sources: List[str]


class StructuredFeedback_v1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    feedback_type: FeedbackType
    target: str = Field(max_length=500)


class AIEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)


def validate_json_structure(data: str, schema_class: Type[BaseModel]) -> AIEvaluationResult:
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        return AIEvaluationResult(success=False, error=f"Invalid JSON: {exc}", confidence=0.0)

    try:
        validated = schema_class.model_validate(parsed)
    except ValidationError as exc:
        return AIEvaluationResult(success=False, error=f"Schema validation failed: {exc}", confidence=0.0)

    confidence = float(getattr(validated, "confidence", 0.8))
    return AIEvaluationResult(success=True, output=validated.model_dump(), confidence=confidence)


def create_fallback_output(task_type: str, confidence: float = 0.2) -> Dict[str, Any]:
    safe_confidence = round(max(0.0, min(float(confidence), 1.0)), 2)

    if task_type == "profiling":
        return {
            "risk_flags": [RiskFlag.NONE.value],
            "dosha_estimate": {"vata": 0.33, "pitta": 0.33, "kapha": 0.34},
            "confidence": safe_confidence,
        }

    if task_type == "feedback":
        return {"feedback_type": FeedbackType.DISLIKE.value, "target": ""}

    if task_type == "explanation":
        return {"explanation": "", "sources": []}

    return {}


class DoshaValidator:
    @staticmethod
    def validate_sum(vata: float, pitta: float, kapha: float) -> bool:
        return abs((vata + pitta + kapha) - 1.0) <= 0.02

    @staticmethod
    def normalize(vata: float, pitta: float, kapha: float) -> Tuple[float, float, float]:
        total = vata + pitta + kapha
        if total <= 0:
            return (0.33, 0.33, 0.34)
        a = round(vata / total, 2)
        b = round(pitta / total, 2)
        c = round(1.0 - a - b, 2)
        return (a, b, c)


SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "profiling": AIProfileOutput_v1,
    "feedback": StructuredFeedback_v1,
    "explanation": StructuredExplanation_v1,
}
