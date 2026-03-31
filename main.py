import asyncio
import json
import os
import re
from typing import Any, Dict, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from llm_prompts_strict import (
    get_feedback_parsing_prompt,
    get_profiling_prompt,
    get_rag_explanation_prompt,
)
from llm_schemas_strict import create_fallback_output
from llm_hallucination_control import ContextValidator

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage
except Exception:  # pragma: no cover
    ChatGroq = None
    HumanMessage = None


app = FastAPI(title="AYUDIET AI Service", version="1.0.0")

LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))


class ProfileRequest(BaseModel):
    text: str = Field(default="")


class ExplainRequest(BaseModel):
    context: str = Field(default="")
    reasoning: str = Field(default="")


class FeedbackRequest(BaseModel):
    text: str = Field(default="")


def ok_response(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"success": True, "data": data, "error": None}


def error_response(error_type: str, message: str) -> Dict[str, Any]:
    return {
        "success": False,
        "data": None,
        "error": {"type": error_type, "message": message},
    }


def _log(prefix: str, payload: Any) -> None:
    print(f"[{prefix}] {payload}")


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


def _build_profile_fallback() -> Dict[str, Any]:
    return create_fallback_output("profiling", 0.3)


def _build_feedback_fallback() -> Dict[str, Any]:
    return {"feedback_type": "DISLIKE", "target": ""}


def _build_explain_fallback(context: str) -> Dict[str, Any]:
    safe_context = (context or "").strip()
    explanation = "Insufficient context to provide explanation."
    if safe_context:
        explanation = safe_context[:200]
    return {"explanation": explanation}


def _get_llm() -> Optional[Any]:
    if ChatGroq is None or HumanMessage is None:
        return None
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    try:
        return ChatGroq(model=model, temperature=0, api_key=api_key)
    except Exception:
        return None


async def _invoke_llm_json(prompt: str) -> Optional[Dict[str, Any]]:
    llm = _get_llm()
    if llm is None:
        raise RuntimeError("LLM unavailable")

    def _call() -> str:
        response = llm.invoke([HumanMessage(content=prompt)])
        return str(response.content).strip()

    raw_output = await asyncio.wait_for(asyncio.to_thread(_call), timeout=LLM_TIMEOUT_SECONDS)
    _log("LLM raw output", raw_output)
    parsed = _extract_json_block(raw_output)
    _log("LLM parsed output", parsed)
    return parsed


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ai/profile")
async def ai_profile(payload: ProfileRequest) -> Dict[str, Any]:
    _log("request received", {"endpoint": "/ai/profile", "text": payload.text})
    fallback = _build_profile_fallback()
    try:
        prompt = get_profiling_prompt(payload.text)
        parsed = await _invoke_llm_json(prompt)
        if not parsed:
            return ok_response(fallback)

        required = {"risk_flags", "dosha_estimate", "confidence"}
        if not required.issubset(parsed.keys()):
            return ok_response(fallback)

        dosha = parsed.get("dosha_estimate", {})
        if not {"vata", "pitta", "kapha"}.issubset(dosha.keys()):
            return ok_response(fallback)

        return ok_response(
            {
                "risk_flags": parsed.get("risk_flags", []),
                "dosha_estimate": {
                    "vata": float(dosha.get("vata", 0.33)),
                    "pitta": float(dosha.get("pitta", 0.33)),
                    "kapha": float(dosha.get("kapha", 0.34)),
                },
                "confidence": float(parsed.get("confidence", 0.3)),
            }
        )
    except asyncio.TimeoutError:
        _log("errors", "profile timeout")
        return ok_response(fallback)
    except Exception as exc:
        _log("errors", str(exc))
        return ok_response(fallback)


@app.post("/ai/explain")
async def ai_explain(payload: ExplainRequest) -> Dict[str, Any]:
    _log(
        "request received",
        {"endpoint": "/ai/explain", "context": payload.context, "reasoning": payload.reasoning},
    )
    fallback = _build_explain_fallback(payload.context)
    try:
        prompt = get_rag_explanation_prompt(payload.context, payload.reasoning)
        parsed = await _invoke_llm_json(prompt)
        if not parsed:
            return ok_response(fallback)

        explanation = str(parsed.get("explanation", "")).strip()
        if not explanation:
            return ok_response(fallback)

        has_external_claims, adherence = ContextValidator.check_context_adherence(
            explanation, [payload.context]
        )
        if has_external_claims:
            _log("errors", f"context adherence low: {adherence}")
            return ok_response(fallback)

        return ok_response({"explanation": explanation})
    except asyncio.TimeoutError:
        _log("errors", "explain timeout")
        return ok_response(fallback)
    except Exception as exc:
        _log("errors", str(exc))
        return ok_response(fallback)


@app.post("/ai/feedback")
async def ai_feedback(payload: FeedbackRequest) -> Dict[str, Any]:
    _log("request received", {"endpoint": "/ai/feedback", "text": payload.text})
    fallback = _build_feedback_fallback()
    try:
        prompt = get_feedback_parsing_prompt(payload.text)
        parsed = await _invoke_llm_json(prompt)
        if not parsed:
            return ok_response(fallback)

        feedback_type = str(parsed.get("feedback_type", "DISLIKE")).upper()
        if feedback_type not in {"LIKE", "DISLIKE", "REPLACE"}:
            feedback_type = "DISLIKE"

        target = str(parsed.get("target", "")).strip()
        return ok_response({"feedback_type": feedback_type, "target": target})
    except asyncio.TimeoutError:
        _log("errors", "feedback timeout")
        return ok_response(fallback)
    except Exception as exc:
        _log("errors", str(exc))
        return ok_response(fallback)
