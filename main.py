from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from llm_hallucination_control import ContextValidator, OutputValidator
from llm_strict_wrapper import StrictLLMWrapper
from schemas import (
    DoshaEstimate,
    Envelope,
    ErrorBody,
    ExplainRequest,
    ExplainResponse,
    HealthData,
    ProfileRequest,
    ProfileResponse,
)

try:
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover
    ChatGroq = None


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


def _parse_allowed_origins() -> list[str]:
    default_dev_origins = [
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


def _request_meta(request: Request) -> tuple[str, str]:
    request_id = request.headers.get("X-Request-ID", "").strip() or "unknown"
    trace_id = request.headers.get("X-Trace-ID", "").strip() or request_id
    return request_id, trace_id


def _success(data: Any) -> Dict[str, Any]:
    if hasattr(data, "model_dump"):
        payload = data.model_dump()
    else:
        payload = data
    return {"success": True, "data": payload, "error": None}


def _error(message: str) -> Dict[str, Any]:
    return {"success": False, "data": None, "error": ErrorBody(message=message).model_dump()}


def _build_profile_fallback() -> ProfileResponse:
    return ProfileResponse(
        symptom_tags=["none"],
        primary_dosha="pitta",
        dosha_estimate=DoshaEstimate(vata=0.33, pitta=0.33, kapha=0.34),
        confidence=0.2,
        fallback=True,
    )


def _build_explain_fallback() -> ExplainResponse:
    return ExplainResponse(
        explanation="insufficient context to provide safe explanation",
        reasoning=["fallback"],
        confidence=0.1,
        sources=[],
        fallback=True,
    )


def _build_ai_explain_data(text: str) -> Dict[str, str]:
    cleaned = _normalize_text(text)
    if not cleaned:
        cleaned = _build_explain_fallback().explanation
    return {"text": cleaned[:2000]}


def _build_rag_data(answer: str, sources: Any) -> Dict[str, Any]:
    cleaned_answer = _normalize_text(answer)
    if not cleaned_answer:
        cleaned_answer = _build_explain_fallback().explanation

    clean_sources: list[str] = []
    if isinstance(sources, list):
        clean_sources = [_normalize_text(src) for src in sources if isinstance(src, str) and src.strip()]

    return {"answer": cleaned_answer[:2000], "sources": clean_sources}


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


def _primary_dosha_from_scores(scores: DoshaEstimate) -> str:
    ordered = ["vata", "pitta", "kapha"]
    values = {"vata": scores.vata, "pitta": scores.pitta, "kapha": scores.kapha}
    return max(ordered, key=lambda name: (values[name], -ordered.index(name)))


def sanitize_ai_output(text: str) -> str:
    forbidden = [
        "meal",
        "recipe",
        "select",
        "choose",
        "recommend food",
        "eat",
        "plan",
        "diet plan",
        "ranking",
        "optimize",
    ]
    cleaned = _normalize_text(text)
    lowered = cleaned.lower()
    for word in forbidden:
        if word in lowered:
            raise ValueError("AI boundary violation")
    return cleaned


def _get_llm() -> Optional[Any]:
    if ChatGroq is None:
        return None
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    try:
        return ChatGroq(model=model, temperature=0, api_key=api_key)
    except Exception:
        return None


def _get_strict_wrapper() -> Optional[StrictLLMWrapper]:
    global _STRICT_WRAPPER
    if _STRICT_WRAPPER is not None:
        return _STRICT_WRAPPER
    llm = _get_llm()
    if llm is None:
        return None
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
            return JSONResponse(status_code=exc.status_code, content=_error(str(exc.detail)))
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
    return JSONResponse(status_code=422, content=_error("Invalid request payload"))


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
    return JSONResponse(status_code=exc.status_code, content=_error(str(exc.detail)))


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
    return JSONResponse(status_code=exc.status_code, content=_error(str(exc.detail)))


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
    return JSONResponse(status_code=500, content=_error("Internal server error"))


@app.get("/health", response_model=Envelope[HealthData])
@app.get("/health/", response_model=Envelope[HealthData])
@app.get("/ai/health", response_model=Envelope[HealthData])
@app.get("/ai/health/", response_model=Envelope[HealthData])
@app.get("/api/health", response_model=Envelope[HealthData])
@app.get("/api/health/", response_model=Envelope[HealthData])
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
    return _success(HealthData(status="ok", entrypoint="main.py"))


@app.post("/profile", response_model=Envelope[ProfileResponse])
@app.post("/profile/", response_model=Envelope[ProfileResponse])
@app.post("/ai/profile", response_model=Envelope[ProfileResponse])
@app.post("/ai/profile/", response_model=Envelope[ProfileResponse])
@app.post("/api/ai/profile", response_model=Envelope[ProfileResponse])
@app.post("/api/ai/profile/", response_model=Envelope[ProfileResponse])
async def profile(payload: ProfileRequest, request: Request) -> Dict[str, Any]:
    request_id, trace_id = _request_meta(request)
    start = time.perf_counter()
    wrapper = _get_strict_wrapper()

    log.info(
        "profile_request",
        extra={
            "request_id": request_id,
            "trace_id": trace_id,
            "endpoint": "/profile",
        },
    )

    if wrapper is None:
        fallback = _build_profile_fallback()
        log.warning(
            "fallback_triggered",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/profile",
                "reason": "LLM unavailable",
            },
        )
        return _success(fallback)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(wrapper.extract_health_profile, payload.symptoms, {}),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        data = result.get("data", {})
        is_valid, parsed, _ = OutputValidator.validate_profile_output(data)
        if not is_valid or parsed is None:
            fallback = _build_profile_fallback()
            log.warning(
                "fallback_triggered",
                extra={
                    "request_id": request_id,
                    "trace_id": trace_id,
                    "endpoint": "/profile",
                    "reason": "schema_validation_failed",
                },
            )
            return _success(fallback)

        profile_data = ProfileResponse(
            symptom_tags=[str(flag) for flag in parsed.risk_flags],
            primary_dosha=_primary_dosha_from_scores(parsed.dosha_estimate),
            dosha_estimate=parsed.dosha_estimate,
            confidence=round(float(parsed.confidence), 2),
            fallback=False,
        )

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
        return _success(profile_data)

    except asyncio.TimeoutError:
        fallback = _build_profile_fallback()
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/profile",
                "reason": "timeout",
            },
        )
        return _success(fallback)
    except Exception as exc:
        fallback = _build_profile_fallback()
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/profile",
                "reason": str(exc),
            },
        )
        return _success(fallback)


@app.post("/explain", response_model=Envelope[ExplainResponse])
@app.post("/explain/", response_model=Envelope[ExplainResponse])
@app.post("/api/explain", response_model=Envelope[ExplainResponse])
@app.post("/api/explain/", response_model=Envelope[ExplainResponse])
async def explain(payload: ExplainRequest, request: Request) -> Dict[str, Any]:
    request_id, trace_id = _request_meta(request)
    start = time.perf_counter()
    wrapper = _get_strict_wrapper()

    log.info(
        "explain_request",
        extra={
            "request_id": request_id,
            "trace_id": trace_id,
            "endpoint": "/explain",
        },
    )

    if wrapper is None:
        fallback = _build_explain_fallback()
        log.warning(
            "fallback_triggered",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/explain",
                "reason": "LLM unavailable",
            },
        )
        return _success(fallback)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(wrapper.generate_explanation, [payload.context], payload.reasoning),
            timeout=LLM_TIMEOUT_SECONDS,
        )

        data = result.get("data", {})
        raw_explanation = str(data.get("explanation", "")).strip()
        sanitized = sanitize_ai_output(raw_explanation)

        has_external_claims, _ = ContextValidator.check_context_adherence(sanitized, [payload.context])
        if has_external_claims:
            raise ValueError("Context adherence violation")

        sources = data.get("sources", [])
        if not isinstance(sources, list) or any(not isinstance(src, str) for src in sources):
            raise ValueError("Invalid sources")

        penalty = float(data.get("confidence_penalty", 0.0))
        confidence = round(max(0.0, min(1.0, 1.0 - penalty)), 2)

        explain_data = ExplainResponse(
            explanation=sanitized,
            reasoning=[_normalize_text(payload.reasoning)],
            confidence=confidence,
            sources=[_normalize_text(src) for src in sources],
            fallback=False,
        )

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
        return _success(explain_data)

    except asyncio.TimeoutError:
        fallback = _build_explain_fallback()
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/explain",
                "reason": "timeout",
            },
        )
        return _success(fallback)
    except ValueError as exc:
        fallback = _build_explain_fallback()
        log.warning(
            "fallback_triggered",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/explain",
                "reason": str(exc),
            },
        )
        return _success(fallback)
    except Exception as exc:
        fallback = _build_explain_fallback()
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/explain",
                "reason": str(exc),
            },
        )
        return _success(fallback)


@app.post("/ai/explain")
@app.post("/ai/explain/")
@app.post("/api/ai/explain")
@app.post("/api/ai/explain/")
async def ai_explain(request: Request) -> Dict[str, Any]:
    request_id, trace_id = _request_meta(request)
    start = time.perf_counter()
    wrapper = _get_strict_wrapper()
    payload = await _safe_json_body(request)
    query = _extract_query(payload)

    log.info(
        "ai_explain_request",
        extra={
            "request_id": request_id,
            "trace_id": trace_id,
            "endpoint": "/ai/explain",
            "has_query": bool(query),
        },
    )

    if not query:
        return JSONResponse(status_code=400, content=_error("query is required"))

    if wrapper is None:
        fallback = _build_explain_fallback()
        log.warning(
            "fallback_triggered",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/ai/explain",
                "reason": "LLM unavailable",
            },
        )
        return _success(_build_ai_explain_data(fallback.explanation))

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                wrapper.generate_explanation,
                [query],
                f"Explain this Ayurvedic query clearly: {query}",
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        data = result.get("data", {})
        raw_explanation = str(data.get("explanation", "")).strip()
        sanitized = sanitize_ai_output(raw_explanation)
        if not sanitized:
            raise ValueError("Empty explanation generated")
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
        return _success(_build_ai_explain_data(text))
    except asyncio.TimeoutError:
        fallback = _build_explain_fallback()
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/ai/explain",
                "reason": "timeout",
            },
        )
        return _success(_build_ai_explain_data(fallback.explanation))
    except ValueError as exc:
        fallback = _build_explain_fallback()
        log.warning(
            "fallback_triggered",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/ai/explain",
                "reason": str(exc),
            },
        )
        return _success(_build_ai_explain_data(fallback.explanation))
    except Exception as exc:
        fallback = _build_explain_fallback()
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/ai/explain",
                "reason": str(exc),
            },
        )
        return _success(_build_ai_explain_data(fallback.explanation))


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
    wrapper = _get_strict_wrapper()
    payload = await _safe_json_body(request)
    query = _extract_query(payload)
    context = _extract_context(payload)

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
        return JSONResponse(status_code=400, content=_error("query is required"))

    if wrapper is None:
        fallback = _build_explain_fallback()
        log.warning(
            "fallback_triggered",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/rag",
                "reason": "LLM unavailable",
            },
        )
        return _success(_build_rag_data(fallback.explanation, fallback.sources))

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                wrapper.generate_explanation,
                [context, query],
                (
                    "Use the provided context first and answer the Ayurvedic query. "
                    f"Context: {context}. Query: {query}"

                ),
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
        data = result.get("data", {})
        raw_answer = str(data.get("explanation", "")).strip()
        answer = sanitize_ai_output(raw_answer)
        if not answer:
            raise ValueError("Empty answer generated")
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
        return _success(_build_rag_data(answer, sources))
    except asyncio.TimeoutError:
        fallback = _build_explain_fallback()
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/rag",
                "reason": "timeout",
            },
        )
        return _success(_build_rag_data(fallback.explanation, fallback.sources))
    except ValueError as exc:
        fallback = _build_explain_fallback()
        log.warning(
            "fallback_triggered",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/rag",
                "reason": str(exc),
            },
        )
        return _success(_build_rag_data(fallback.explanation, fallback.sources))
    except Exception as exc:
        fallback = _build_explain_fallback()
        log.error(
            "ai_failure",
            extra={
                "request_id": request_id,
                "trace_id": trace_id,
                "endpoint": "/rag",
                "reason": str(exc),
            },
        )
        return _success(_build_rag_data(fallback.explanation, fallback.sources))
