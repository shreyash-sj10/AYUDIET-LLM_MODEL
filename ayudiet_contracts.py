from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


RulePriority = Literal["P0", "P1", "P2", "P3"]


class UserState_v1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    demographics: Dict[str, Any] = Field(default_factory=dict)
    conditions: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    symptoms: List[str] = Field(default_factory=list)
    prakriti: Dict[str, Any] = Field(default_factory=dict)
    goal: str = "maintenance"
    template_id: str = "lunch_roti_dal_sabzi"
    hard_constraints: List[str] = Field(default_factory=list)
    soft_preferences: List[str] = Field(default_factory=list)


class TemplateSlot_v1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot: str
    required: bool
    allowed_categories: List[str]


class TemplatePlan_v1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_id: str
    slots: List[TemplateSlot_v1]
    fixed_components: Dict[str, str] = Field(default_factory=dict)
    flexible_components: List[str] = Field(default_factory=list)


class ExclusionTraceItem_v1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot: str
    food_id: str
    rule: str
    priority: RulePriority


class ScoreBreakdown_v1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_alignment: float = 0.0
    dosha_alignment: float = 0.0
    digestibility: float = 0.0
    symptom_support: float = 0.0
    preference_match: float = 0.0
    diversity_adjustment: float = 0.0


class CandidateScore_v1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    food_id: str
    slot: str
    score_total: float
    score_breakdown: ScoreBreakdown_v1


class Trace_v1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules_version: str
    cache_key: str
    exclusions: List[ExclusionTraceItem_v1] = Field(default_factory=list)
    scored_candidates: List[CandidateScore_v1] = Field(default_factory=list)
    optimization_steps: List[str] = Field(default_factory=list)
    relaxations_applied: List[RulePriority] = Field(default_factory=list)


class DecisionResponse_v1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    template_id: str
    meal: Dict[str, Optional[str]]
    relaxation_level_used: Optional[RulePriority] = None
    fallback_used: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    trace: Trace_v1
