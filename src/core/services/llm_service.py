"""Pooled Groq clients for latency-sensitive, in-interview LLM calls.

Final holistic evaluation is isolated in ``evaluation_llm_client`` and uses
NVIDIA NIM instead of this module.
"""

from __future__ import annotations

import logging
import time
from typing import Any, TypeVar

from groq import AsyncGroq
from pydantic import BaseModel

from src.config.settings import settings

logger = logging.getLogger(__name__)

_clients: dict[tuple[str, bool], AsyncGroq] = {}
ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


def _clean_api_key(value: str | None) -> str:
    return str(value or "").strip().strip('"').strip("'")


def _select_api_key(*, use_fallback: bool, purpose: str) -> str:
    key_pairs = {
        "classification": (
            settings.GROQ_CLASSIFICATION_KEY,
            settings.FALLBACK_GROQ_CLASSIFICATION_KEY,
        ),
        "evaluation": (
            settings.GROQ_EVALUATION_KEY,
            settings.FALLBACK_GROQ_EVALUATION_KEY,
        ),
        "question": (
            settings.GROQ_QUESTION_GENERATION_KEY,
            settings.FALLBACK_GROQ_QUESTION_GENERATION_KEY,
        ),
    }
    try:
        configured_primary, configured_fallback = key_pairs[purpose]
    except KeyError as exc:
        raise ValueError(f"Unsupported Groq client purpose: {purpose}") from exc

    primary = _clean_api_key(configured_primary)
    fallback = _clean_api_key(configured_fallback)

    key = fallback if use_fallback else primary or fallback
    if not key:
        raise RuntimeError(f"Groq {purpose} API key is not configured.")
    return key


def _get_groq_client(purpose: str, use_fallback: bool = False) -> AsyncGroq:
    """Return (and lazily create) the singleton primary or fallback AsyncGroq client."""
    cache_key = (purpose, use_fallback)
    if cache_key not in _clients:
        api_key = _select_api_key(use_fallback=use_fallback, purpose=purpose)
        timeouts = {
            "classification": settings.GROQ_CLASSIFY_TIMEOUT_SECS,
            "evaluation": settings.GROQ_LIVE_EVAL_TIMEOUT_SECS,
            "question": settings.GROQ_QUESTION_TIMEOUT_SECS,
        }
        _clients[cache_key] = AsyncGroq(
            api_key=api_key,
            timeout=timeouts[purpose],
            max_retries=0,
        )
    return _clients[cache_key]


def _structured_response_format(
    response_model: type[BaseModel],
) -> dict[str, Any]:
    """Build Groq strict Structured Outputs configuration from a Pydantic model."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__.lower(),
            "strict": True,
            "schema": response_model.model_json_schema(),
        },
    }


async def _call_with_fallback(
    purpose: str,
    response_model: type[ResponseModelT],
    **kwargs: Any,
) -> ResponseModelT:
    """Execute one strict-schema call and fail over immediately to a backup key."""

    started_at = time.perf_counter()
    provider = "primary"
    kwargs["reasoning_effort"] = settings.GROQ_REASONING_EFFORT
    kwargs["response_format"] = _structured_response_format(response_model)
    try:
        client = _get_groq_client(purpose, use_fallback=False)
        completion = await client.chat.completions.create(**kwargs)
    except Exception as exc:
        primary_key = _select_api_key(use_fallback=False, purpose=purpose)
        fallback_key = _select_api_key(use_fallback=True, purpose=purpose)
        if not fallback_key or fallback_key == primary_key:
            raise
        logger.warning(
            "Groq call failed; switching immediately to fallback",
            extra={
                "purpose": purpose,
                "provider": "primary",
                "error_type": type(exc).__name__,
                "status_code": getattr(exc, "status_code", None),
                "elapsed_ms": round(
                    (time.perf_counter() - started_at) * 1000,
                    2,
                ),
            },
        )
        provider = "fallback"
        client = _get_groq_client(purpose, use_fallback=True)
        completion = await client.chat.completions.create(**kwargs)

    logger.info(
        "Groq call completed",
        extra={
            "purpose": purpose,
            "provider": provider,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
        },
    )
    raw_content = completion.choices[0].message.content or ""
    return response_model.model_validate_json(raw_content)


async def classify(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
) -> ResponseModelT:
    """Classify one candidate response through strict Structured Outputs."""
    return await _call_with_fallback(
        purpose="classification",
        response_model=response_model,
        model=settings.GROQ_CLASSIFY_MODEL,
        messages=messages,
        max_tokens=settings.GROQ_CLASSIFY_MAX_TOKENS,
        temperature=settings.GROQ_CLASSIFY_TEMPERATURE,
    )


async def live_evaluate(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
) -> ResponseModelT:
    """Evaluate one substantial answer through strict Structured Outputs."""
    return await _call_with_fallback(
        purpose="evaluation",
        response_model=response_model,
        model=settings.GROQ_LIVE_EVAL_MODEL,
        messages=messages,
        max_tokens=settings.GROQ_LIVE_EVAL_MAX_TOKENS,
        temperature=settings.GROQ_LIVE_EVAL_TEMPERATURE,
    )


async def generate(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
) -> ResponseModelT:
    """Generate one question through strict Structured Outputs."""
    return await _call_with_fallback(
        purpose="question",
        response_model=response_model,
        model=settings.GROQ_QUESTION_MODEL,
        messages=messages,
        max_tokens=settings.GROQ_QUESTION_MAX_TOKENS,
        temperature=settings.GROQ_QUESTION_TEMPERATURE,
    )


async def lightweight(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
    *,
    max_tokens: int = 192,
    temperature: float = 0.3,
) -> ResponseModelT:
    """Run a short classifier-model completion through strict Structured Outputs."""
    return await _call_with_fallback(
        purpose="classification",
        response_model=response_model,
        model=settings.GROQ_CLASSIFY_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
