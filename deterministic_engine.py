from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ayudiet_contracts import (
    CandidateScore_v1,
    DecisionResponse_v1,
    ExclusionTraceItem_v1,
    ScoreBreakdown_v1,
    TemplatePlan_v1,
    TemplateSlot_v1,
    Trace_v1,
    UserState_v1,
)


RULES_VERSION = "v1.0.0"


@dataclass
class FoodRecord:
    food_id: str
    slot_category: str
    calories: float
    protein: float
    digestibility: float
    heaviness: float
    dosha_effect: Dict[str, float]


class DeterministicDecisionEngine:
    """Constraint-first deterministic meal recommendation pipeline."""

    def __init__(self) -> None:
        self._cache: Dict[str, DecisionResponse_v1] = {}
        self._history_by_session: Dict[str, List[str]] = {}

    def recommend_meal(
        self,
        *,
        session_id: str,
        user_profile: Dict[str, Any],
        query: str,
        datasets: Dict[str, pd.DataFrame],
        template_id: str = "lunch_roti_dal_sabzi",
    ) -> DecisionResponse_v1:
        user_state = self._build_user_state(user_profile, query, template_id)
        template_plan = self._build_template_plan(template_id)
        catalog = self._build_food_catalog(datasets)

        cache_key = self._cache_key(user_state, template_plan)
        if cache_key in self._cache:
            return deepcopy(self._cache[cache_key])

        relax_order = [None, "P3", "P2", "P1"]
        all_exclusions: List[ExclusionTraceItem_v1] = []
        final_candidates_by_slot: Dict[str, List[FoodRecord]] = {}
        relaxations_applied: List[str] = []
        relaxation_level_used: Optional[str] = None

        for relax in relax_order:
            candidates_by_slot, exclusions = self._apply_constraints(
                user_state=user_state,
                template_plan=template_plan,
                catalog=catalog,
                relax_until=relax,
            )
            all_exclusions.extend(exclusions)

            if all(candidates_by_slot.get(slot.slot) for slot in template_plan.slots if slot.required):
                final_candidates_by_slot = candidates_by_slot
                relaxation_level_used = relax
                if relax is not None:
                    relaxations_applied.append(relax)
                break

        if not final_candidates_by_slot:
            response = self._fallback_response(template_plan, cache_key)
            self._cache[cache_key] = response
            return deepcopy(response)

        scored = self._score_candidates(user_state, final_candidates_by_slot)
        adjusted = self._apply_diversity(session_id, scored)
        meal, optimization_steps = self._optimize(template_plan, adjusted)

        if not meal:
            response = self._fallback_response(template_plan, cache_key)
            self._cache[cache_key] = response
            return deepcopy(response)

        self._history_by_session.setdefault(session_id, [])
        self._history_by_session[session_id].extend([v for v in meal.values() if v])
        self._history_by_session[session_id] = self._history_by_session[session_id][-30:]

        confidence = self._confidence_score(final_candidates_by_slot, relaxations_applied, fallback_used=False)
        response = DecisionResponse_v1(
            template_id=template_plan.template_id,
            meal=meal,
            relaxation_level_used=relaxation_level_used,
            fallback_used=False,
            confidence=confidence,
            trace=Trace_v1(
                rules_version=RULES_VERSION,
                cache_key=cache_key,
                exclusions=all_exclusions,
                scored_candidates=adjusted,
                optimization_steps=optimization_steps,
                relaxations_applied=relaxations_applied,
            ),
        )
        self._cache[cache_key] = response
        return deepcopy(response)

    def _build_user_state(self, user_profile: Dict[str, Any], query: str, template_id: str) -> UserState_v1:
        profile = user_profile or {}
        symptoms = []
        text = f"{query} {' '.join(profile.get('health_conditions', []))}".lower()
        for token in ["bloating", "fatigue", "acidity", "constipation", "burning"]:
            if token in text:
                symptoms.append(token)

        conditions = [str(c).lower() for c in profile.get("health_conditions", [])]
        allergies = [str(a).lower() for a in profile.get("allergies", [])]
        goals = profile.get("health_goals", [])
        goal = str(goals[0]).lower() if goals else "maintenance"

        dosha_scores = profile.get("dosha_scores", {})
        if not dosha_scores:
            primary = str(profile.get("primary_dosha", "pitta")).lower()
            dosha_scores = {"vata": 0.33, "pitta": 0.33, "kapha": 0.34}
            if primary in dosha_scores:
                dosha_scores = {"vata": 0.2, "pitta": 0.6, "kapha": 0.2} if primary == "pitta" else {"vata": 0.6, "pitta": 0.2, "kapha": 0.2} if primary == "vata" else {"vata": 0.2, "pitta": 0.2, "kapha": 0.6}

        soft_prefs = [f"prefer_{p}" for p in profile.get("dietary_preferences", [])]
        soft_prefs.extend([f"avoid_{d}" for d in profile.get("food_dislikes", [])])

        return UserState_v1(
            demographics={
                "age": profile.get("age"),
                "gender": profile.get("gender"),
                "weight": profile.get("weight"),
                "height": profile.get("height"),
            },
            conditions=conditions,
            allergies=allergies,
            symptoms=symptoms,
            prakriti={"source": "assistive", "distribution": dosha_scores, "confidence": 0.7},
            goal=goal,
            template_id=template_id,
            hard_constraints=["P0_allergy_exclusion", "P0_template_integrity", "P1_condition_alignment"],
            soft_preferences=soft_prefs,
        )

    def _build_template_plan(self, template_id: str) -> TemplatePlan_v1:
        templates = {
            "lunch_roti_dal_sabzi": TemplatePlan_v1(
                template_id="lunch_roti_dal_sabzi",
                slots=[
                    TemplateSlot_v1(slot="grain", required=True, allowed_categories=["grain"]),
                    TemplateSlot_v1(slot="dal", required=True, allowed_categories=["dal"]),
                    TemplateSlot_v1(slot="sabzi", required=True, allowed_categories=["vegetable"]),
                    TemplateSlot_v1(slot="side", required=False, allowed_categories=["side"]),
                ],
                fixed_components={},
                flexible_components=["grain", "dal", "sabzi", "side"],
            )
        }
        return templates.get(template_id, templates["lunch_roti_dal_sabzi"])

    def _infer_category(self, name: str) -> str:
        n = name.lower()
        if any(k in n for k in ["dal", "lentil", "moong", "masoor", "toor", "rajma", "chana"]):
            return "dal"
        if any(k in n for k in ["roti", "rice", "wheat", "jowar", "bajra", "millet", "phulka"]):
            return "grain"
        if any(k in n for k in ["curd", "raita", "salad", "buttermilk"]):
            return "side"
        return "vegetable"

    def _build_food_catalog(self, datasets: Dict[str, pd.DataFrame]) -> List[FoodRecord]:
        df = None
        for key in ["indb", "nin_fct", "uk_fct", "us_fct", "indb_ayurvedic"]:
            candidate = datasets.get(key)
            if candidate is not None and not candidate.empty and "food_name" in candidate.columns:
                df = candidate
                break

        if df is None or df.empty:
            return [
                FoodRecord("jowar_roti", "grain", 120, 4, 0.85, 0.2, {"vata": 0.0, "pitta": 0.1, "kapha": -0.2}),
                FoodRecord("moong_dal", "dal", 105, 7, 0.9, 0.2, {"vata": 0.0, "pitta": -0.1, "kapha": -0.2}),
                FoodRecord("lauki_sabzi", "vegetable", 60, 2, 0.92, 0.1, {"vata": 0.0, "pitta": -0.1, "kapha": -0.1}),
                FoodRecord("salad", "side", 35, 1, 0.8, 0.1, {"vata": 0.1, "pitta": -0.1, "kapha": -0.1}),
            ]

        records: List[FoodRecord] = []
        sample = df.head(2500)
        for _, row in sample.iterrows():
            name = str(row.get("food_name", "")).strip()
            if not name:
                continue
            food_id = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_")
            calories = float(row.get("energy_kcal", 100) or 100)
            protein = float(row.get("protein_g", 3) or 3)
            digestibility = 0.8 if calories < 180 else 0.6 if calories < 280 else 0.4
            heaviness = min(1.0, calories / 350)
            category = self._infer_category(name)
            dosha = {"vata": 0.0, "pitta": 0.0, "kapha": -0.1 if calories < 180 else 0.1}
            records.append(FoodRecord(food_id, category, calories, protein, digestibility, heaviness, dosha))

        return records

    def _is_allergy_match(self, food_id: str, allergies: List[str]) -> bool:
        return any(a and a in food_id for a in allergies)

    def _condition_violation(self, food: FoodRecord, conditions: List[str]) -> bool:
        if "pcos" in conditions or "diabetes" in conditions:
            return food.calories > 260
        return False

    def _symptom_violation(self, food: FoodRecord, symptoms: List[str]) -> bool:
        if "bloating" in symptoms and food.heaviness > 0.65:
            return True
        if "acidity" in symptoms and any(k in food.food_id for k in ["fried", "spicy"]):
            return True
        return False

    def _pref_dislike_violation(self, food: FoodRecord, prefs: List[str]) -> bool:
        avoid_tokens = [p.replace("avoid_", "") for p in prefs if p.startswith("avoid_")]
        return any(t and t in food.food_id for t in avoid_tokens)

    def _apply_constraints(
        self,
        *,
        user_state: UserState_v1,
        template_plan: TemplatePlan_v1,
        catalog: List[FoodRecord],
        relax_until: Optional[str],
    ) -> Tuple[Dict[str, List[FoodRecord]], List[ExclusionTraceItem_v1]]:
        allowed_relax = {"P3": {"P3"}, "P2": {"P3", "P2"}, "P1": {"P3", "P2", "P1"}, None: set()}
        relax_set = allowed_relax.get(relax_until, set())

        by_slot: Dict[str, List[FoodRecord]] = {slot.slot: [] for slot in template_plan.slots}
        exclusions: List[ExclusionTraceItem_v1] = []

        for slot in template_plan.slots:
            for food in catalog:
                if food.slot_category not in slot.allowed_categories:
                    continue

                excluded = False
                if self._is_allergy_match(food.food_id, user_state.allergies):
                    exclusions.append(ExclusionTraceItem_v1(slot=slot.slot, food_id=food.food_id, rule="allergy_exclusion", priority="P0"))
                    excluded = True

                if not excluded and "P1" not in relax_set and self._condition_violation(food, user_state.conditions):
                    exclusions.append(ExclusionTraceItem_v1(slot=slot.slot, food_id=food.food_id, rule="condition_rule", priority="P1"))
                    excluded = True

                if not excluded and "P2" not in relax_set and self._symptom_violation(food, user_state.symptoms):
                    exclusions.append(ExclusionTraceItem_v1(slot=slot.slot, food_id=food.food_id, rule="symptom_rule", priority="P2"))
                    excluded = True

                if not excluded and "P3" not in relax_set and self._pref_dislike_violation(food, user_state.soft_preferences):
                    exclusions.append(ExclusionTraceItem_v1(slot=slot.slot, food_id=food.food_id, rule="preference_dislike", priority="P3"))
                    excluded = True

                if not excluded:
                    by_slot[slot.slot].append(food)

        return by_slot, exclusions

    def _score_candidates(
        self,
        user_state: UserState_v1,
        candidates_by_slot: Dict[str, List[FoodRecord]],
    ) -> List[CandidateScore_v1]:
        scores: List[CandidateScore_v1] = []
        dosha = user_state.prakriti.get("distribution", {"vata": 0.33, "pitta": 0.33, "kapha": 0.34})

        for slot, foods in candidates_by_slot.items():
            for food in foods:
                goal_alignment = 0.25 if (user_state.goal == "weight_loss" and food.calories < 220) else 0.15
                dosha_alignment = 0.15 + max(
                    (food.dosha_effect.get("vata", 0) * float(dosha.get("vata", 0.33))),
                    (food.dosha_effect.get("pitta", 0) * float(dosha.get("pitta", 0.33))),
                    (food.dosha_effect.get("kapha", 0) * float(dosha.get("kapha", 0.34))),
                )
                digestibility = 0.25 * food.digestibility
                symptom_support = 0.18 if not self._symptom_violation(food, user_state.symptoms) else 0.04
                preference_match = 0.12 if any(pref.replace("prefer_", "") in food.food_id for pref in user_state.soft_preferences if pref.startswith("prefer_")) else 0.05

                breakdown = ScoreBreakdown_v1(
                    goal_alignment=round(goal_alignment, 3),
                    dosha_alignment=round(max(0.0, dosha_alignment), 3),
                    digestibility=round(digestibility, 3),
                    symptom_support=round(symptom_support, 3),
                    preference_match=round(preference_match, 3),
                    diversity_adjustment=0.0,
                )
                total = (
                    breakdown.goal_alignment
                    + breakdown.dosha_alignment
                    + breakdown.digestibility
                    + breakdown.symptom_support
                    + breakdown.preference_match
                )
                scores.append(
                    CandidateScore_v1(
                        food_id=food.food_id,
                        slot=slot,
                        score_total=round(total, 4),
                        score_breakdown=breakdown,
                    )
                )

        scores.sort(key=lambda s: (-s.score_total, s.food_id))
        return scores

    def _apply_diversity(self, session_id: str, scored: List[CandidateScore_v1]) -> List[CandidateScore_v1]:
        history = self._history_by_session.get(session_id, [])
        adjusted: List[CandidateScore_v1] = []
        for item in scored:
            penalty = 0.08 if item.food_id in history[-10:] else 0.0
            item.score_breakdown.diversity_adjustment = round(-penalty, 3)
            item.score_total = round(max(0.0, item.score_total - penalty), 4)
            adjusted.append(item)

        adjusted.sort(key=lambda s: (-s.score_total, s.food_id))
        return adjusted

    def _optimize(
        self,
        template_plan: TemplatePlan_v1,
        adjusted_scores: List[CandidateScore_v1],
    ) -> Tuple[Dict[str, Optional[str]], List[str]]:
        by_slot: Dict[str, List[CandidateScore_v1]] = {}
        for score in adjusted_scores:
            by_slot.setdefault(score.slot, []).append(score)

        meal: Dict[str, Optional[str]] = {}
        steps: List[str] = []
        used: set[str] = set()

        for slot in template_plan.slots:
            ranked = by_slot.get(slot.slot, [])
            chosen = None
            for candidate in ranked:
                if candidate.food_id not in used:
                    chosen = candidate
                    break
            if chosen is None and slot.required:
                return {}, steps + [f"slot_{slot.slot}_infeasible"]
            if chosen is not None:
                meal[slot.slot] = chosen.food_id
                used.add(chosen.food_id)
                steps.append(f"slot_{slot.slot}_selected:{chosen.food_id}")
            else:
                meal[slot.slot] = None
                steps.append(f"slot_{slot.slot}_optional_empty")

        return meal, steps

    def _fallback_response(self, template_plan: TemplatePlan_v1, cache_key: str) -> DecisionResponse_v1:
        fallback_meals = {
            "lunch_roti_dal_sabzi": {
                "grain": "jowar_roti",
                "dal": "moong_dal",
                "sabzi": "lauki_sabzi",
                "side": None,
            }
        }
        meal = fallback_meals.get(template_plan.template_id, fallback_meals["lunch_roti_dal_sabzi"])
        return DecisionResponse_v1(
            template_id=template_plan.template_id,
            meal=meal,
            relaxation_level_used="P1",
            fallback_used=True,
            confidence=0.58,
            trace=Trace_v1(
                rules_version=RULES_VERSION,
                cache_key=cache_key,
                exclusions=[],
                scored_candidates=[],
                optimization_steps=["fallback_used"],
                relaxations_applied=["P3", "P2", "P1"],
            ),
        )

    def _cache_key(self, user_state: UserState_v1, template_plan: TemplatePlan_v1) -> str:
        key_payload = {
            "user_state": user_state.model_dump(),
            "template": template_plan.model_dump(),
            "rules_version": RULES_VERSION,
        }
        raw = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _confidence_score(
        self,
        candidates_by_slot: Dict[str, List[FoodRecord]],
        relaxations_applied: List[str],
        fallback_used: bool,
    ) -> float:
        breadth = sum(len(v) for v in candidates_by_slot.values())
        base = 0.95 if breadth >= 20 else 0.88 if breadth >= 10 else 0.8
        penalty = 0.1 * len(relaxations_applied)
        if fallback_used:
            penalty += 0.2
        return round(max(0.4, min(0.99, base - penalty)), 2)
