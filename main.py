from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from llm_hallucination_control import ContextValidator, OutputValidator
from llm_strict_wrapper import StrictLLMWrapper
from schemas import (
    DoshaEstimate,
    ExplainRequest,
    ProfileRequest,
)

try:
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover
    ChatGroq = None

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("Missing GROQ_API_KEY")
print("LLM INITIALIZED SUCCESSFULLY")


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("ayudiet.api")

app = FastAPI(title="AYUDIET AI Service", version="2.0.0")

LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))
API_KEY = os.getenv("AYUDIET_API_KEY", "").strip()
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
REQUIRE_AUTH_ON_COMPAT_ENDPOINTS = (
    os.getenv("REQUIRE_AUTH_ON_COMPAT_ENDPOINTS", "false").strip().lower() == "true"
)
_RATE_BUCKETS: Dict[str, Deque[float]] = {}
_STRICT_WRAPPER: Optional[StrictLLMWrapper] = None
_CHAT_MEMORY: Dict[str, Deque[Dict[str, str]]] = {}
_CORE_PROTECTED_PATHS = {
    "/profile",
    "/explain",
    "/profile/",
    "/explain/",
    "/ai/profile",
    "/ai/profile/",
    "/api/ai/profile",
    "/api/ai/profile/",
}
_COMPAT_PROTECTED_PATHS = {
    "/ai/explain",
    "/ai/explain/",
    "/rag",
    "/rag/",
    "/ai/rag",
    "/ai/rag/",
    "/api/rag",
    "/api/rag/",
    "/api/ai/explain",
    "/api/ai/explain/",
    "/api/ai/rag",
    "/api/ai/rag/",
}

_FOLLOWUP_PHRASES = {
    "why",
    "how",
    "what does that mean",
    "then what",
}
_GREETING_PHRASES = {
    "hi",
    "hello",
    "helo",
    "hey",
}


def _parse_allowed_origins() -> list[str]:
    default_dev_origins = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:8011",
        "http://127.0.0.1:8011",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    raw = os.getenv("CORS_ORIGINS", "")
    env_origins = [o.strip() for o in raw.split(",") if o.strip() and o.strip() != "*"]

    # Always allow local frontend origins for development/debugging, even when
    # CORS_ORIGINS is set in Render env vars.
    merged = list(dict.fromkeys(default_dev_origins + env_origins))
    return merged or default_dev_origins


app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
@app.get("/ui")
def serve_ui() -> FileResponse:
    return FileResponse("index.html")


def _request_meta(request: Request) -> tuple[str, str]:
    request_id = request.headers.get("X-Request-ID", "").strip() or "unknown"
    trace_id = request.headers.get("X-Trace-ID", "").strip() or request_id
    return request_id, trace_id


class SafetyBlockedError(ValueError):
    pass


class AdherenceError(ValueError):
    pass


class NoContextError(ValueError):
    pass


def _build_meta(*, fallback: bool, reason: Optional[str], mode: str, retryable: bool) -> Dict[str, Any]:
    return {"fallback": fallback, "reason": reason, "mode": mode, "retryable": retryable}


def _success(
    data: Dict[str, Any],
    *,
    fallback: bool,
    reason: Optional[str],
    mode: str,
    retryable: bool,
) -> Dict[str, Any]:
    return {"success": True, "data": data, "error": None, "meta": _build_meta(fallback=fallback, reason=reason, mode=mode, retryable=retryable)}


def _error(
    message: str,
    *,
    mode: str,
    fallback: bool = False,
    reason: Optional[str] = None,
    retryable: bool = False,
) -> Dict[str, Any]:
    return {"success": False, "data": None, "error": message, "meta": _build_meta(fallback=fallback, reason=reason, mode=mode, retryable=retryable)}


def _build_profile_fallback_data() -> Dict[str, Any]:
    return {
        "text": "Insufficient context to infer a reliable dosha profile.",
        "profile": {
            "primary_dosha": "pitta",
            "symptom_tags": ["none"],
            "confidence": 0.2,
            "dosha_estimate": {"vata": 0.33, "pitta": 0.33, "kapha": 0.34},
        },
    }


def _build_explain_fallback_text() -> str:
    return "insufficient context to provide safe explanation"


def _build_text_data(text: str, **extra: Any) -> Dict[str, Any]:
    cleaned = _normalize_text(text)
    if not cleaned:
        cleaned = _build_explain_fallback_text()
    payload: Dict[str, Any] = {"text": cleaned[:2000]}
    payload.update(extra)
    return payload


def _build_rag_data(answer: str, sources: Any) -> Dict[str, Any]:
    cleaned_answer = _normalize_text(answer)
    if not cleaned_answer:
        cleaned_answer = _build_explain_fallback_text()

    clean_sources: list[str] = []
    if isinstance(sources, list):
        clean_sources = [_normalize_text(src) for src in sources if isinstance(src, str) and src.strip()]

    return {"text": cleaned_answer[:2000], "sources": clean_sources}


def _normalize_text(value: str) -> str:
    return " ".join(str(value).strip().split())


async def _safe_json_body(request: Request) -> Dict[str, Any]:
    try:
        body = await request.json()
        if isinstance(body, dict):
            return body
    except json.JSONDecodeError:
        return {}
    except Exception:
        return {}
    return {}


def _extract_query(payload: Dict[str, Any]) -> str:
    keys = ("query", "question", "prompt", "message")
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            normalized = _normalize_text(value)
            if normalized:
                return normalized
    return ""


def _extract_context(payload: Dict[str, Any]) -> str:
    value = payload.get("context")
    if isinstance(value, str):
        normalized = _normalize_text(value)
        if normalized:
            return normalized
    return "ayurvedic diet"


def _extract_explain_query(payload: Dict[str, Any]) -> str:
    query = _extract_query(payload)
    if query:
        return query

    context = payload.get("context")
    if isinstance(context, str):
        normalized = _normalize_text(context)
        if normalized:
            return normalized
    if isinstance(context, dict):
        parts = []
        for key in ("user_profile", "selected_meal", "constraints_applied", "trace"):
            value = context.get(key)
            if value:
                parts.append(f"{key}: {value}")
        text = _normalize_text(" | ".join(str(part) for part in parts))
        if text:
            return text
    return ""


def _extract_session_id(payload: Dict[str, Any]) -> str:
    session_id = payload.get("session_id")
    if isinstance(session_id, str):
        normalized = _normalize_text(session_id)
        if normalized:
            return normalized
    return "default"


def _extract_chat_history(payload: Dict[str, Any], session_id: str) -> list[str]:
    history: list[str] = []

    external = payload.get("chat_history")
    if isinstance(external, list):
        for item in external[-8:]:
            if isinstance(item, str):
                text = _normalize_text(item)
                if text:
                    history.append(text)
            elif isinstance(item, dict):
                role = _normalize_text(item.get("role", ""))
                content = _normalize_text(item.get("content", ""))
                if content:
                    history.append(f"{role or 'message'}: {content}")

    memory = _CHAT_MEMORY.get(session_id)
    if memory:
        for turn in list(memory)[-8:]:
            role = _normalize_text(turn.get("role", "message"))
            content = _normalize_text(turn.get("content", ""))
            if content:
                history.append(f"{role}: {content}")

    return history[-8:]


def _remember_turn(session_id: str, role: str, content: str) -> None:
    if not content.strip():
        return
    memory = _CHAT_MEMORY.setdefault(session_id, deque(maxlen=20))
    memory.append({"role": role, "content": content[:2000]})


def _is_followup_query(query: str) -> bool:
    sample = _normalize_text(query).lower()
    return sample in _FOLLOWUP_PHRASES


def _is_greeting(query: str) -> bool:
    sample = _normalize_text(query).lower()
    return sample in _GREETING_PHRASES


def _primary_dosha_from_scores(scores: DoshaEstimate) -> str:
    ordered = ["vata", "pitta", "kapha"]
    values = {"vata": scores.vata, "pitta": scores.pitta, "kapha": scores.kapha}
    return max(ordered, key=lambda name: (values[name], -ordered.index(name)))


def sanitize_ai_output(text: str) -> str:
    forbidden = [
        "give me a diet plan",
        "what should i eat",
        "suggest meals",
    ]
    cleaned = _normalize_text(text)
    lowered = cleaned.lower()
    for word in forbidden:
        if word in lowered:
            raise SafetyBlockedError("safety_blocked")
    return cleaned


def _get_llm() -> Any:
    if ChatGroq is None:
        raise RuntimeError("Missing GROQ_API_KEY")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    try:
        return ChatGroq(model=model, temperature=0, api_key=api_key)
    except Exception as exc:
        raise RuntimeError("Missing GROQ_API_KEY") from exc


def _get_strict_wrapper() -> StrictLLMWrapper:
    global _STRICT_WRAPPER
    if _STRICT_WRAPPER is not None:
        return _STRICT_WRAPPER
    llm = _get_llm()
    _STRICT_WRAPPER = StrictLLMWrapper(llm)
    return _STRICT_WRAPPER


def _enforce_api_key(request: Request) -> None:
    if not API_KEY:
        return
    provided = request.headers.get("X-API-Key", "").strip()
    if not provided:
        auth_header = request.headers.get("Authorization", "").strip()
        if auth_header.lower().startswith("bearer "):
            provided = auth_header[7:].strip()
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _enforce_rate_limit(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _RATE_BUCKETS.setdefault(ip, deque())
    while bucket and now - bucket[0] > 60:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_MINUTE:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    bucket.append(now)


def _mode_from_path(path: str) -> str:
    lowered = (path or "").lower()
    if "profile" in lowered:
        return "profile"
    if "rag" in lowered:
        return "rag"
    return "explain"


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    protected_paths = set(_CORE_PROTECTED_PATHS)
    if REQUIRE_AUTH_ON_COMPAT_ENDPOINTS:
        protected_paths.update(_COMPAT_PROTECTED_PATHS)

    if request.url.path in protected_paths:
        try:
            _enforce_api_key(request)
            _enforce_rate_limit(request)
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content=_error(str(exc.detail), mode=_mode_from_path(request.url.path), reason="safety_blocked"),
            )
    return await call_next(request)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id, trace_id = _request_meta(request)
    log.warning(
        "validation_error",
        extra={
            "request_id": request_id,
            "trace_id": trace_id,
            "endpoint": request.url.path,
            "errors": exc.errors(),
        },
    )
    return JSONResponse(
        status_code=422,
        content=_error("Invalid request payload", mode=_mode_from_path(request.url.path), reason="no_context"),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id, trace_id = _request_meta(request)
    log.warning(
        "http_error",
        extra={
            "request_id": request_id,
            "trace_id": trace_id,
            "endpoint": request.url.path,
            "status_code": exc.status_code,
            "detail": str(exc.detail),
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_error(str(exc.detail), mode=_mode_from_path(request.url.path), reason="safety_blocked"),
    )


@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
    request_id, trace_id = _request_meta(request)
    log.warning(
        "http_error",
        extra={
            "request_id": request_id,
            "trace_id": trace_id,
            "endpoint": request.url.path,
            "status_code": exc.status_code,
            "detail": str(exc.detail),
        },
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=_error(str(exc.detail), mode=_mode_from_path(request.url.path), reason="safety_blocked"),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    request_id, trace_id = _request_meta(request)
    log.error(
        "runtime_error",
        extra={
            "request_id": request_id,
            "trace_id": trace_id,
            "endpoint": request.url.path,
            "error": str(exc),
        },
    )
    return JSONResponse(
        status_code=500,
        content=_error(
            "Internal server error",
            mode=_mode_from_path(request.url.path),
            reason="timeout",
            retryable=True,
        ),
    )


@app.get("/health")
@app.get("/health/")
@app.get("/ai/health")
@app.get("/ai/health/")
@app.get("/api/health")
@app.get("/api/health/")
async def health(request: Request) -> Dict[str, Any]:
    request_id, trace_id = _request_meta(request)
    log.info(
        "health_request",
        extra={
            "request_id": request_id,
            "trace_id": trace_id,
            "endpoint": "/health",
        },
    )
    return _success(
        {"text": "Service is healthy", "status": "ok", "entrypoint": "main.py"},
        fallback=False,
        reason=None,
        mode="explain",
        retryable=False,
    )


@app.post("/profile")
@app.post("/profile/")
@app.post("/ai/profile")
@app.post("/ai/profile/")
@app.post("/api/ai/profile")
@app.post("/api/ai/profile/")
async def profile(payload: ProfileRequest, request: Request) -> Dict[str, Any]:
    request_id, trace_id = _request_meta(request)
    start = time.perf_counter()
    try:
        wrapper = _get_strict_wrapper()
    except Exception:
        return _success(
            _build_profile_fallback_data(),
            fallback=True,
            reason="llm_not_initialized",
            mode="profile",
            retryable=True,
        )

    log.info(
        "profile_request",
        extra={
            "request_id": request_id,
            "trace_id": trace_id,
            "endpoint": "/profile",
        },
    )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(wrapper.extract_health_profile, payload.symptoms, {}),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        data = result.get("data", {})
        is_valid, parsed, _ = OutputValidator.validate_profile_output(data)
        if not is_valid or parsed is None:
            fallback = _build_profile_fallback_data()
            log.warning(
                "fallback_triggered",
                extra={
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "endpoint": "/profile",
                    "reason": "schema_validation_failed",
                },
            )
            return _success(fallback, fallback=True, reason="no_context", mode="profile", retryable=False)

        dosha_raw = parsed.dosha_estimate.model_dump() if hasattr(parsed.dosha_estimate, "model_dump") else parsed.dosha_estimate
        dosha_scores = DoshaEstimate.model_validate(dosha_raw)
        normalized_tags: list[str] = []
        for flag in parsed.risk_flags:
            raw = str(flag)
            if "." in raw:
                raw = raw.split(".")[-1]
            normalized_tags.append(raw.lower().replace(" ", "_"))

        primary = _primary_dosha_from_scores(dosha_scores)
        profile_data = {
            "text": f"User shows signs of {primary} imbalance with symptom-linked indicators.",
            "profile": {
                "primary_dosha": primary,
                "symptom_tags": normalized_tags or ["none"],
                "confidence": round(float(parsed.confidence), 2),
                "dosha_estimate": dosha_scores.model_dump(),
            },
        }

        latency_ms = int((time.perf_counter() - start) * 1000)
        log.info(
            "profile_response",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/profile",
                "latency_ms": latency_ms,
                "fallback": False,
            },
        )
        return _success(profile_data, fallback=False, reason="profile_generated", mode="profile", retryable=False)

    except asyncio.TimeoutError:
        fallback = _build_profile_fallback_data()
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/profile",
                "reason": "timeout",
            },
        )
        return _success(fallback, fallback=True, reason="timeout", mode="profile", retryable=True)
    except Exception as exc:
        fallback = _build_profile_fallback_data()
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/profile",
                "reason": str(exc),
            },
        )
        reason = "llm_not_initialized" if "Missing GROQ_API_KEY" in str(exc) else "timeout"
        return _success(fallback, fallback=True, reason=reason, mode="profile", retryable=True)


@app.post("/explain")
@app.post("/explain/")
@app.post("/api/explain")
@app.post("/api/explain/")
async def explain(payload: ExplainRequest, request: Request) -> Dict[str, Any]:
    request_id, trace_id = _request_meta(request)
    start = time.perf_counter()
    try:
        wrapper = _get_strict_wrapper()
    except Exception:
        return _success(
            _build_text_data(_build_explain_fallback_text(), sources=[]),
            fallback=True,
            reason="llm_not_initialized",
            mode="explain",
            retryable=True,
        )

    log.info(
        "explain_request",
        extra={
            "request_id": request_id,
            "trace_id": trace_id,
            "endpoint": "/explain",
        },
    )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(wrapper.generate_explanation, [payload.context], payload.reasoning),
            timeout=LLM_TIMEOUT_SECONDS,
        )

        data = result.get("data", {})
        raw_explanation = str(data.get("explanation", "")).strip()
        sanitized = sanitize_ai_output(raw_explanation)
        if not sanitized:
            raise NoContextError("no_context")

        has_external_claims, _ = ContextValidator.check_context_adherence(sanitized, [payload.context])
        if has_external_claims:
            raise AdherenceError("adherence_failed")

        sources = data.get("sources", [])
        if not isinstance(sources, list) or any(not isinstance(src, str) for src in sources):
            raise NoContextError("no_context")

        explain_data = _build_text_data(sanitized, sources=[_normalize_text(src) for src in sources])

        latency_ms = int((time.perf_counter() - start) * 1000)
        log.info(
            "explain_response",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/explain",
                "latency_ms": latency_ms,
                "fallback": False,
            },
        )
        return _success(explain_data, fallback=False, reason="explanation_generated", mode="explain", retryable=False)

    except asyncio.TimeoutError:
        fallback = _build_text_data(_build_explain_fallback_text())
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/explain",
                "reason": "timeout",
            },
        )
        return _success(fallback, fallback=True, reason="timeout", mode="explain", retryable=True)
    except SafetyBlockedError:
        fallback = _build_text_data(_build_explain_fallback_text())
        log.warning(
            "fallback_triggered",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/explain",
                "reason": "safety_blocked",
            },
        )
        return _success(fallback, fallback=True, reason="safety_blocked", mode="explain", retryable=False)
    except AdherenceError:
        fallback = _build_text_data(_build_explain_fallback_text())
        return _success(fallback, fallback=True, reason="adherence_failed", mode="explain", retryable=False)
    except NoContextError:
        fallback = _build_text_data(_build_explain_fallback_text())
        return _success(fallback, fallback=True, reason="no_context", mode="explain", retryable=False)
    except Exception as exc:
        fallback = _build_text_data(_build_explain_fallback_text())
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/explain",
                "reason": str(exc),
            },
        )
        reason = "llm_not_initialized" if "Missing GROQ_API_KEY" in str(exc) else "timeout"
        return _success(fallback, fallback=True, reason=reason, mode="explain", retryable=True)


@app.post("/ai/explain")
@app.post("/ai/explain/")
@app.post("/api/ai/explain")
@app.post("/api/ai/explain/")
async def ai_explain(request: Request) -> Dict[str, Any]:
    request_id, trace_id = _request_meta(request)
    start = time.perf_counter()
    try:
        wrapper = _get_strict_wrapper()
    except Exception:
        return _success(
            _build_text_data(_build_explain_fallback_text()),
            fallback=True,
            reason="llm_not_initialized",
            mode="explain",
            retryable=True,
        )
    payload = await _safe_json_body(request)
    session_id = _extract_session_id(payload)
    chat_history = _extract_chat_history(payload, session_id)
    query = _extract_explain_query(payload)
    if not query:
        return JSONResponse(
            status_code=400,
            content=_error("query (min 3 chars) is required", mode="explain", reason="no_context"),
        )

    log.info(
        "ai_explain_request",
        extra={
            "request_id": request_id,
            "trace_id": trace_id,
            "endpoint": "/ai/explain",
            "has_query": bool(query),
        },
    )

    try:
        convo_context = "\n".join(chat_history[-6:]) if chat_history else "No prior conversation."
        result = await asyncio.wait_for(
            asyncio.to_thread(
                wrapper.generate_explanation,
                [query],
                f"Conversation Context:\n{convo_context}\n\nCurrent Question:\n{query}",
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        data = result.get("data", {})
        raw_explanation = str(data.get("explanation", "")).strip()
        sanitized = sanitize_ai_output(raw_explanation)
        if not sanitized:
            safe_result = await asyncio.wait_for(
                asyncio.to_thread(
                    wrapper.generate_safe_text,
                    f"Explain this safely and clearly: {query}",
                    _build_explain_fallback_text(),
                    max_chars=2000,
                ),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            safe_data = safe_result.get("data", {})
            sanitized = sanitize_ai_output(str(safe_data.get("response", "")).strip())
            if not sanitized:
                raise NoContextError("no_context")
        text = sanitized[:2000]
        latency_ms = int((time.perf_counter() - start) * 1000)
        log.info(
            "ai_explain_response",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/ai/explain",
                "latency_ms": latency_ms,
                "fallback": False,
            },
        )
        _remember_turn(session_id, "user", query)
        _remember_turn(session_id, "assistant", text)
        return _success(_build_text_data(text), fallback=False, reason="explanation_generated", mode="explain", retryable=False)
    except asyncio.TimeoutError:
        fallback = _build_text_data(_build_explain_fallback_text())
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/ai/explain",
                "reason": "timeout",
            },
        )
        return _success(fallback, fallback=True, reason="timeout", mode="explain", retryable=True)
    except SafetyBlockedError:
        fallback = _build_text_data(_build_explain_fallback_text())
        log.warning(
            "fallback_triggered",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/ai/explain",
                "reason": "safety_blocked",
            },
        )
        return _success(fallback, fallback=True, reason="safety_blocked", mode="explain", retryable=False)
    except NoContextError:
        fallback = _build_text_data(_build_explain_fallback_text())
        return _success(fallback, fallback=True, reason="no_context", mode="explain", retryable=False)
    except Exception as exc:
        fallback = _build_text_data(_build_explain_fallback_text())
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/ai/explain",
                "reason": str(exc),
            },
        )
        reason = "llm_not_initialized" if "Missing GROQ_API_KEY" in str(exc) else "timeout"
        return _success(fallback, fallback=True, reason=reason, mode="explain", retryable=True)


@app.post("/rag")
@app.post("/rag/")
@app.post("/ai/rag")
@app.post("/ai/rag/")
@app.post("/api/rag")
@app.post("/api/rag/")
@app.post("/api/ai/rag")
@app.post("/api/ai/rag/")
async def rag(request: Request) -> Dict[str, Any]:
    request_id, trace_id = _request_meta(request)
    start = time.perf_counter()
    try:
        wrapper = _get_strict_wrapper()
    except Exception:
        return _success(
            _build_text_data(_build_explain_fallback_text(), sources=[]),
            fallback=True,
            reason="llm_not_initialized",
            mode="rag",
            retryable=True,
        )
    payload = await _safe_json_body(request)
    session_id = _extract_session_id(payload)
    chat_history = _extract_chat_history(payload, session_id)
    query = _extract_query(payload)
    context = _extract_context(payload)
    retrieved_documents = payload.get("retrieved_documents")

    log.info(
        "rag_request",
        extra={
            "request_id": request_id,
            "trace_id": trace_id,
            "endpoint": "/rag",
            "has_query": bool(query),
        },
    )

    if not query:
        return JSONResponse(status_code=400, content=_error("query is required", mode="rag", reason="no_context"))

    if _is_greeting(query):
        greeting_text = (
            "Hello! I can help you understand Ayurveda, your symptoms, or explain your diet plan. "
            "What would you like to know?"
        )
        _remember_turn(session_id, "user", query)
        _remember_turn(session_id, "assistant", greeting_text)
        return _success(
            _build_text_data(greeting_text, sources=["conversation"]),
            fallback=False,
            reason="rag_answer_generated",
            mode="rag",
            retryable=False,
        )

    try:
        is_followup = _is_followup_query(query)
        convo_context = "\n".join(chat_history[-6:]) if chat_history else "No prior conversation."
        context_chunks = [context, query]
        allow_general_knowledge = False

        if isinstance(retrieved_documents, list):
            docs = [_normalize_text(str(doc)) for doc in retrieved_documents if str(doc).strip()]
            if docs:
                context_chunks = docs + [query]
            else:
                allow_general_knowledge = True
        elif retrieved_documents is None:
            allow_general_knowledge = True
        else:
            normalized_doc = _normalize_text(str(retrieved_documents))
            if normalized_doc:
                context_chunks = [normalized_doc, query]
            else:
                allow_general_knowledge = True

        if allow_general_knowledge:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    wrapper.generate_safe_text,
                    (
                        "You are a conversational Ayurveda assistant. Answer naturally in 3-6 sentences.\n"
                        f"Conversation Context:\n{convo_context}\n\n"
                        f"Current Question:\n{query}\n\n"
                        "Answer style:\n"
                        "1) direct answer\n"
                        "2) simple explanation\n"
                        "3) practical meaning\n"
                        "Avoid specific diet plans or food prescriptions."
                    ),
                    _build_explain_fallback_text(),
                    max_chars=2000,
                ),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            data = result.get("data", {})
            raw_answer = str(data.get("response", "")).strip()
            answer = sanitize_ai_output(raw_answer)
            if not answer:
                raise NoContextError("no_context")
            sources = ["general_knowledge"]
        else:
            followup_instruction = (
                "This is a follow-up question. Use the previous assistant response for continuity."
                if is_followup
                else "Answer based on available conversation context."
            )
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    wrapper.generate_explanation,
                    context_chunks + [f"Conversation Context: {convo_context}"],
                    (
                        "Use provided documents/context first.\n"
                        f"{followup_instruction}\n"
                        "Structure: direct answer -> simple explanation -> practical meaning.\n"
                        "Target 3-6 sentences.\n"
                        "Avoid specific diet plans or food prescriptions.\n"
                        f"Current Question: {query}"
                    ),
                ),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            data = result.get("data", {})
            raw_answer = str(data.get("explanation", "")).strip()
            answer = sanitize_ai_output(raw_answer)
            if not answer:
                raise NoContextError("no_context")
            sources = data.get("sources", [])
            if not isinstance(sources, list):
                sources = []

        latency_ms = int((time.perf_counter() - start) * 1000)
        log.info(
            "rag_response",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/rag",
                "latency_ms": latency_ms,
                "fallback": False,
            },
        )
        _remember_turn(session_id, "user", query)
        _remember_turn(session_id, "assistant", answer)
        return _success(_build_rag_data(answer, sources), fallback=False, reason="rag_answer_generated", mode="rag", retryable=False)
    except asyncio.TimeoutError:
        fallback = _build_text_data(_build_explain_fallback_text(), sources=[])
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/rag",
                "reason": "timeout",
            },
        )
        return _success(fallback, fallback=True, reason="timeout", mode="rag", retryable=True)
    except SafetyBlockedError:
        fallback = _build_text_data(_build_explain_fallback_text(), sources=[])
        log.warning(
            "fallback_triggered",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/rag",
                "reason": "safety_blocked",
            },
        )
        return _success(fallback, fallback=True, reason="safety_blocked", mode="rag", retryable=False)
    except NoContextError:
        fallback = _build_text_data(_build_explain_fallback_text(), sources=[])
        return _success(fallback, fallback=True, reason="no_context", mode="rag", retryable=False)
    except Exception as exc:
        fallback = _build_text_data(_build_explain_fallback_text(), sources=[])
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/rag",
                "reason": str(exc),
            },
        )
        reason = "llm_not_initialized" if "Missing GROQ_API_KEY" in str(exc) else "timeout"
        return _success(fallback, fallback=True, reason=reason, mode="rag", retryable=True)


def run_system_tests():
    import requests

    base = "http://localhost:8000"

    tests = [
        {
            "name": "GENERAL KNOWLEDGE",
            "endpoint": "/ai/rag",
            "payload": {"query": "What is Ayurveda?"},
        },
        {
            "name": "EXPLAIN",
            "endpoint": "/ai/explain",
            "payload": {
                "type": "EXPLAIN_DECISION",
                "context": {
                    "user_profile": {"dosha": "pitta"},
                    "selected_meal": {"items": ["rice"]},
                    "constraints_applied": ["no spicy"],
                    "trace": "cooling foods selected",
                },
            },
        },
        {
            "name": "SYMPTOMS",
            "endpoint": "/ai/profile",
            "payload": {"query": "I feel acidity and heat"},
        },
    ]

    for test in tests:
        try:
            response = requests.post(base + test["endpoint"], json=test["payload"], timeout=30)
            print("\nTEST:", test["name"])
            print("STATUS:", response.status_code)
            print("RESPONSE:", response.json())
        except Exception as exc:
            print("ERROR:", str(exc))


def run_final_tests():
    import requests

    BASE = "http://localhost:8000"

    tests = [
        {
            "name": "GENERAL",
            "endpoint": "/ai/rag",
            "payload": {"query": "What is Ayurveda?"},
        },
        {
            "name": "EXPLAIN",
            "endpoint": "/ai/explain",
            "payload": {
                "type": "EXPLAIN_DECISION",
                "context": {
                    "user_profile": {"dosha": "pitta"},
                    "selected_meal": {"items": ["rice"]},
                    "constraints_applied": ["no spicy"],
                    "trace": "cooling foods selected",
                },
            },
        },
        {
            "name": "PROFILE",
            "endpoint": "/ai/profile",
            "payload": {"query": "I feel acidity and heat"},
        },
    ]

    for test in tests:
        try:
            response = requests.post(BASE + test["endpoint"], json=test["payload"], timeout=45)
            print("\nTEST:", test["name"])
            print(response.json())
        except Exception as exc:
            print("ERROR:", str(exc))

    print("LLM SERVICE READY FOR DEPLOYMENT")
