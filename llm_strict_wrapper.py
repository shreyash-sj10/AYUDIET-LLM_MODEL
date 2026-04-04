import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from llm_confidence_calibration import ConfidenceCalibrator, ConfidenceThresholds
from llm_hallucination_control import ContextValidator, FallbackGenerator, OutputValidator
from llm_prompts_strict import (
    build_safe_context,
    get_feedback_parsing_prompt,
    get_profiling_prompt,
    get_rag_explanation_prompt,
)


class StrictLLMWrapper:
    def __init__(self, llm_instance: Any):
        self.llm = llm_instance

    def _invoke_json(self, prompt: str) -> Optional[Dict[str, Any]]:
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self._extract_json_block(str(response.content).strip())

    def call(
        self,
        prompt: str,
        *,
        fallback: Dict[str, Any],
        validator: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Generic strict call path.
        Enforces JSON extraction and optional validator-based schema checks.
        """
        payload = self._invoke_json(prompt)
        if payload is None:
            return {"success": False, "data": fallback}

        if validator is None:
            return {"success": True, "data": payload}

        is_valid, parsed, _ = validator(payload)
        if not is_valid or parsed is None:
            return {"success": False, "data": fallback}
        return {"success": True, "data": parsed.model_dump()}

    @staticmethod
    def _extract_json_block(raw_text: str) -> Optional[Dict[str, Any]]:
        if not raw_text:
            return None
        candidate = raw_text.strip()
        try:
            return json.loads(candidate)
        except Exception:
            pass

        fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", candidate, re.IGNORECASE)
        if fenced:
            try:
                return json.loads(fenced.group(1).strip())
            except Exception:
                pass

        first = candidate.find("{")
        last = candidate.rfind("}")
        if first != -1 and last != -1 and last > first:
            try:
                return json.loads(candidate[first : last + 1])
            except Exception:
                return None
        return None

    def extract_health_profile(self, user_input: str, current_profile: Optional[Dict] = None) -> Dict[str, Any]:
        safe_input = build_safe_context(user_input)
        if not safe_input:
            return {"success": False, "data": FallbackGenerator.create_fallback_profile(0.1)}

        pre_confidence = ConfidenceCalibrator.get_confidence_score(safe_input)
        if ConfidenceThresholds.should_use_fallback(pre_confidence):
            return {"success": False, "data": FallbackGenerator.create_fallback_profile(pre_confidence)}

        payload = self._invoke_json(get_profiling_prompt(safe_input, current_profile or {}))
        if payload is None:
            return {"success": False, "data": FallbackGenerator.create_fallback_profile(min(pre_confidence, 0.3))}

        is_valid, parsed, _ = OutputValidator.validate_profile_output(payload)
        if not is_valid or parsed is None:
            return {"success": False, "data": FallbackGenerator.create_fallback_profile(min(pre_confidence, 0.3))}

        ambiguity = ConfidenceCalibrator.detect_ambiguity(safe_input)
        calibrated = ConfidenceCalibrator.calibrate_confidence(parsed.confidence, ambiguity)
        data = parsed.model_dump()
        data["confidence"] = calibrated
        return {"success": True, "data": data}

    def parse_feedback(self, feedback_text: str) -> Dict[str, Any]:
        safe_input = build_safe_context(feedback_text)
        if not safe_input:
            return {"success": False, "data": FallbackGenerator.create_fallback_feedback()}

        payload = self._invoke_json(get_feedback_parsing_prompt(safe_input))
        if payload is None:
            return {"success": False, "data": FallbackGenerator.create_fallback_feedback()}

        is_valid, parsed, _ = OutputValidator.validate_feedback_output(payload)
        if not is_valid or parsed is None:
            return {"success": False, "data": FallbackGenerator.create_fallback_feedback()}

        return {"success": True, "data": parsed.model_dump()}

    def generate_explanation(self, context_chunks: List[str], decision_trace: str) -> Dict[str, Any]:
        if not context_chunks:
            return {"success": False, "data": FallbackGenerator.create_fallback_explanation()}

        safe_context = "\n\n".join(str(chunk)[:700] for chunk in context_chunks[:5])
        safe_trace = build_safe_context(decision_trace, max_length=500)
        payload = self._invoke_json(get_rag_explanation_prompt(safe_context, safe_trace))
        if payload is None:
            return {"success": False, "data": FallbackGenerator.create_fallback_explanation()}

        is_valid, parsed, _ = OutputValidator.validate_explanation_output(payload)
        if not is_valid or parsed is None:
            return {"success": False, "data": FallbackGenerator.create_fallback_explanation()}

        has_external, adherence = ContextValidator.check_context_adherence(parsed.explanation, context_chunks)
        penalty = ContextValidator.confidence_penalty(adherence)
        if has_external:
            return {
                "success": False,
                "data": FallbackGenerator.create_fallback_explanation(),
                "adherence": round(adherence, 2),
                "confidence_penalty": penalty,
            }

        result_data = parsed.model_dump()
        result_data["confidence_penalty"] = penalty
        return {"success": True, "data": result_data, "adherence": round(adherence, 2)}

    def generate_safe_text(
        self,
        instruction: str,
        fallback_text: str,
        *,
        max_chars: int = 6000,
    ) -> Dict[str, Any]:
        """
        Force free-text generation through strict JSON envelope:
        {"response": "<text>", "confidence": <0..1>}
        """
        safe_instruction = build_safe_context(instruction, max_length=3500)
        prompt = f"""Return ONLY JSON:
{{
  "response": "string",
  "confidence": number (0-1)
}}

Rules:
- No markdown wrappers
- No extra fields
- Keep response factual and concise

Instruction:
{safe_instruction}
"""
        payload = self._invoke_json(prompt)
        if payload is None:
            return {"success": False, "data": {"response": fallback_text, "confidence": 0.2}}

        response_text = str(payload.get("response", "")).strip()
        if not response_text:
            return {"success": False, "data": {"response": fallback_text, "confidence": 0.2}}

        try:
            confidence = float(payload.get("confidence", 0.3))
        except Exception:
            confidence = 0.3
        confidence = max(0.0, min(1.0, confidence))

        return {
            "success": True,
            "data": {"response": response_text[:max_chars], "confidence": round(confidence, 2)},
        }
