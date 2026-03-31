import json
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
        try:
            return json.loads(str(response.content).strip())
        except Exception:
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
        if has_external:
            return {
                "success": False,
                "data": FallbackGenerator.create_fallback_explanation(),
                "adherence": round(adherence, 2),
            }

        return {"success": True, "data": parsed.model_dump(), "adherence": round(adherence, 2)}
