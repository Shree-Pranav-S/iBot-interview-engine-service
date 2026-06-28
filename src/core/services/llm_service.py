"""Pooled Groq clients for latency-sensitive, in-interview LLM calls.

Final holistic evaluation is isolated in ``evaluation_llm_client`` and uses
NVIDIA NIM instead of this module.
"""

from __future__ import annotations

import logging

from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_fixed

from src.config.settings import settings

logger = logging.getLogger(__name__)

# â”€â”€ Singleton client â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_client: AsyncGroq | None = None
_fallback_client: AsyncGroq | None = None
_question_client: AsyncGroq | None = None
_question_fallback_client: AsyncGroq | None = None
_evaluation_client: AsyncGroq | None = None
_evaluation_fallback_client: AsyncGroq | None = None


def _clean_api_key(value: str | None) -> str:
    return str(value or "").strip().strip('"').strip("'")


def _select_api_key(*, use_fallback: bool, purpose: str) -> str:
    if purpose == "evaluation" and settings.GROQ_EVALUATION_API_KEY:
        primary = _clean_api_key(settings.GROQ_EVALUATION_API_KEY)
    elif purpose == "question" and settings.GROQ_QUESTION_API_KEY:
        primary = _clean_api_key(settings.GROQ_QUESTION_API_KEY)
    else:
        primary = _clean_api_key(settings.GROQ_API_KEY)

    fallback = _clean_api_key(settings.FALLBACK_GROQ_API_KEY)
    key = (fallback or primary) if use_fallback else (primary or fallback)

    if not key:
        raise RuntimeError(
            f"Groq {purpose} call requires GROQ_API_KEY, or FALLBACK_GROQ_API_KEY"
        )
    return key


def _get_client(use_fallback: bool = False) -> AsyncGroq:
    """Return (and lazily create) the singleton primary or fallback AsyncGroq client."""
    global _client, _fallback_client
    api_key = _select_api_key(use_fallback=use_fallback, purpose="live")
    if use_fallback:
        if _fallback_client is None:
            _fallback_client = AsyncGroq(api_key=api_key, timeout=10.0)
        return _fallback_client

    if _client is None:
        _client = AsyncGroq(api_key=api_key, timeout=10.0)
    return _client


# Typed helpers


def _get_question_client(use_fallback: bool = False) -> AsyncGroq:
    """Return a pooled Groq client with the question-generation timeout."""

    global _question_client, _question_fallback_client
    api_key = _select_api_key(use_fallback=use_fallback, purpose="question")
    if use_fallback:
        if _question_fallback_client is None:
            _question_fallback_client = AsyncGroq(
                api_key=api_key,
                timeout=settings.GROQ_QUESTION_TIMEOUT_SECS,
            )
        return _question_fallback_client

    if _question_client is None:
        _question_client = AsyncGroq(
            api_key=api_key,
            timeout=settings.GROQ_QUESTION_TIMEOUT_SECS,
        )
    return _question_client


def _get_evaluation_client(use_fallback: bool = False) -> AsyncGroq:
    """Return a pooled Groq client for live evaluation."""

    global _evaluation_client, _evaluation_fallback_client
    api_key = _select_api_key(use_fallback=use_fallback, purpose="evaluation")
    if use_fallback:
        if _evaluation_fallback_client is None:
            _evaluation_fallback_client = AsyncGroq(api_key=api_key, timeout=10.0)
        return _evaluation_fallback_client

    if _evaluation_client is None:
        _evaluation_client = AsyncGroq(api_key=api_key, timeout=10.0)
    return _evaluation_client


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def classify(messages: list[dict]) -> str:
    """Call the fast classification model (8B). Returns raw text."""
    try:
        client = _get_client(use_fallback=False)
        completion = await client.chat.completions.create(
            model=settings.GROQ_CLASSIFY_MODEL,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=settings.GROQ_CLASSIFY_MAX_TOKENS,
            temperature=settings.GROQ_CLASSIFY_TEMPERATURE,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.warning(
            "Groq classify failed with primary API key: %s. Retrying with fallback...",
            e,
        )
        client = _get_client(use_fallback=True)
        completion = await client.chat.completions.create(
            model=settings.GROQ_CLASSIFY_MODEL,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=settings.GROQ_CLASSIFY_MAX_TOKENS,
            temperature=settings.GROQ_CLASSIFY_TEMPERATURE,
            response_format={"type": "json_object"},
        )
    return completion.choices[0].message.content or ""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def live_evaluate(messages: list[dict]) -> str:
    """Call the fast live technical evaluation model with JSON output."""
    try:
        client = _get_evaluation_client(use_fallback=False)
        completion = await client.chat.completions.create(
            model=settings.GROQ_LIVE_EVAL_MODEL,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=settings.GROQ_LIVE_EVAL_MAX_TOKENS,
            temperature=settings.GROQ_LIVE_EVAL_TEMPERATURE,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.warning(
            "Groq live_evaluate failed with primary API key: %s. Retrying with fallback...",
            e,
        )
        client = _get_evaluation_client(use_fallback=True)
        completion = await client.chat.completions.create(
            model=settings.GROQ_LIVE_EVAL_MODEL,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=settings.GROQ_LIVE_EVAL_MAX_TOKENS,
            temperature=settings.GROQ_LIVE_EVAL_TEMPERATURE,
            response_format={"type": "json_object"},
        )
    return completion.choices[0].message.content or ""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def generate(messages: list[dict]) -> str:
    """Call the question generation model with JSON output. Returns raw JSON string."""
    try:
        client = _get_question_client(use_fallback=False)
        completion = await client.chat.completions.create(
            model=settings.GROQ_QUESTION_MODEL,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=settings.GROQ_QUESTION_MAX_TOKENS,
            temperature=settings.GROQ_QUESTION_TEMPERATURE,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.warning(
            "Groq generate failed with primary API key: %s. Retrying with fallback...",
            e,
        )
        client = _get_question_client(use_fallback=True)
        completion = await client.chat.completions.create(
            model=settings.GROQ_QUESTION_MODEL,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=settings.GROQ_QUESTION_MAX_TOKENS,
            temperature=settings.GROQ_QUESTION_TEMPERATURE,
            response_format={"type": "json_object"},
        )
    return completion.choices[0].message.content or ""


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
async def lightweight(
    messages: list[dict],
    *,
    max_tokens: int = 192,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> str:
    """Call the fast 8B model for short completions (rephrase, transition, closing, etc.)."""
    kwargs: dict = {
        "model": settings.GROQ_CLASSIFY_MODEL,
        "messages": messages,  # type: ignore[arg-type]
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        client = _get_client(use_fallback=False)
        completion = await client.chat.completions.create(**kwargs)
    except Exception as e:
        logger.warning(
            "Groq lightweight failed with primary API key: %s. Retrying with fallback...",
            e,
        )
        client = _get_client(use_fallback=True)
        completion = await client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content or ""
