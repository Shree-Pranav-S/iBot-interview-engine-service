"""Pooled Groq clients for latency-sensitive, in-interview LLM calls.

Final holistic evaluation is isolated in ``evaluation_llm_client`` and uses
NVIDIA NIM instead of this module.
"""

from __future__ import annotations

import logging
from typing import Any

from groq import AsyncGroq

from src.config.settings import settings

logger = logging.getLogger(__name__)

_clients: dict[tuple[str, bool], AsyncGroq] = {}


def _clean_api_key(value: str | None) -> str:
    return str(value or "").strip().strip('"').strip("'")


def _select_api_key(*, use_fallback: bool, purpose: str) -> str:
    keys = {
        "evaluation": settings.GROQ_EVALUATION_API_KEY,
        "question": settings.GROQ_QUESTION_API_KEY,
        "live": settings.GROQ_API_KEY,
    }
    primary = _clean_api_key(keys.get(purpose) or settings.GROQ_API_KEY)
    fallback = _clean_api_key(settings.FALLBACK_GROQ_API_KEY)

    key = fallback if use_fallback else primary or fallback
    if not key:
        raise RuntimeError(
            f"Groq {purpose} call requires GROQ_API_KEY, or FALLBACK_GROQ_API_KEY"
        )
    return key


def _get_groq_client(purpose: str, use_fallback: bool = False) -> AsyncGroq:
    """Return (and lazily create) the singleton primary or fallback AsyncGroq client."""
    cache_key = (purpose, use_fallback)
    if cache_key not in _clients:
        api_key = _select_api_key(use_fallback=use_fallback, purpose=purpose)
        timeout = settings.GROQ_QUESTION_TIMEOUT_SECS if purpose == "question" else 10.0
        _clients[cache_key] = AsyncGroq(api_key=api_key, timeout=timeout)
    return _clients[cache_key]


async def _call_with_fallback(purpose: str, **kwargs: Any) -> str:
    """Execute once with the primary key and fail over to a distinct backup."""

    try:
        client = _get_groq_client(purpose, use_fallback=False)
        completion = await client.chat.completions.create(**kwargs)
    except Exception as exc:
        primary_key = _select_api_key(use_fallback=False, purpose=purpose)
        fallback_key = _clean_api_key(settings.FALLBACK_GROQ_API_KEY)
        if not fallback_key or fallback_key == primary_key:
            raise
        logger.warning(
            "Groq '%s' failed with the primary key; trying the fallback: %s",
            purpose,
            exc,
        )
        client = _get_groq_client(purpose, use_fallback=True)
        completion = await client.chat.completions.create(**kwargs)

    return completion.choices[0].message.content or ""


async def classify(messages: list[dict]) -> str:
    """Call the fast classification model (8B). Returns raw text."""
    return await _call_with_fallback(
        purpose="live",
        model=settings.GROQ_CLASSIFY_MODEL,
        messages=messages,
        max_tokens=settings.GROQ_CLASSIFY_MAX_TOKENS,
        temperature=settings.GROQ_CLASSIFY_TEMPERATURE,
        response_format={"type": "json_object"},
    )


async def live_evaluate(messages: list[dict]) -> str:
    """Call the fast live technical evaluation model with JSON output."""
    return await _call_with_fallback(
        purpose="evaluation",
        model=settings.GROQ_LIVE_EVAL_MODEL,
        messages=messages,
        max_tokens=settings.GROQ_LIVE_EVAL_MAX_TOKENS,
        temperature=settings.GROQ_LIVE_EVAL_TEMPERATURE,
        response_format={"type": "json_object"},
    )


async def generate(messages: list[dict]) -> str:
    """Call the question generation model with JSON output. Returns raw JSON string."""
    return await _call_with_fallback(
        purpose="question",
        model=settings.GROQ_QUESTION_MODEL,
        messages=messages,
        max_tokens=settings.GROQ_QUESTION_MAX_TOKENS,
        temperature=settings.GROQ_QUESTION_TEMPERATURE,
        response_format={"type": "json_object"},
    )


async def lightweight(
    messages: list[dict],
    *,
    max_tokens: int = 192,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> str:
    """Call the fast 8B model for short completions (rephrase, transition, closing, etc.)."""
    kwargs: dict[str, Any] = {
        "model": settings.GROQ_CLASSIFY_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    return await _call_with_fallback(purpose="live", **kwargs)
