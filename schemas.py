from __future__ import annotations

from typing import Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DoshaEstimate(StrictModel):
    vata: float = Field(ge=0.0, le=1.0)
    pitta: float = Field(ge=0.0, le=1.0)
    kapha: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_sum(self) -> "DoshaEstimate":
        total = round(float(self.vata + self.pitta + self.kapha), 2)
        if abs(total - 1.0) > 0.02:
            raise ValueError("dosha_estimate must sum to 1.0 (+/- 0.02)")
        return self


class ProfileRequest(StrictModel):
    symptoms: str = Field(..., min_length=3)


class ExplainRequest(StrictModel):
    context: str = Field(..., min_length=3)
    reasoning: str = Field(..., min_length=3)


class AIExplainRequest(StrictModel):
    query: str = Field(..., min_length=3)


class RagRequest(StrictModel):
    query: str = Field(..., min_length=3)
    context: str = Field(..., min_length=1)


class HealthData(StrictModel):
    status: Literal["ok"]
    entrypoint: Literal["main.py"]


class ProfileResponse(StrictModel):
    symptom_tags: List[str] = Field(..., min_length=1)
    primary_dosha: Literal["vata", "pitta", "kapha"]
    dosha_estimate: DoshaEstimate
    confidence: float = Field(ge=0.0, le=1.0)
    fallback: bool


class ExplainResponse(StrictModel):
    explanation: str = Field(..., min_length=1, max_length=2000)
    reasoning: List[str] = Field(..., min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[str]
    fallback: bool


class AIExplainData(StrictModel):
    text: str = Field(..., min_length=1, max_length=2000)


class RagData(StrictModel):
    answer: str = Field(..., min_length=1, max_length=2000)
    sources: List[str]


class ErrorBody(StrictModel):
    message: str


T = TypeVar("T", bound=StrictModel)


class Envelope(StrictModel, Generic[T]):
    success: bool
    data: Optional[T]
    error: Optional[ErrorBody]
