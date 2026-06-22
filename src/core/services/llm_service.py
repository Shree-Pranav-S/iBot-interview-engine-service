"""
LLM Service — Centralised Groq client for all interview graph nodes.

Provides a singleton AsyncGroq client (reused across the entire application
lifecycle) and typed helper methods for each LLM task:
  - classify()     → fast 8B model, low tokens
  - evaluate()     → 70B model, structured JSON
  - generate()     → 70B model, question generation
  - lightweight()  → fast 8B model, short free-form completions

Reusing a single client avoids per-call HTTP connection-pool creation,
saving ~100-300ms per graph turn.
"""

from __future__ import annotations

import logging

from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_fixed

from src.config.settings import settings

logger = logging.getLogger(__name__)

# ── Singleton client ──────────────────────────────────────────────────────────

_client: AsyncGroq | None = None
_fallback_client: AsyncGroq | None = None


def _get_client(use_fallback: bool = False) -> AsyncGroq:
    """Return (and lazily create) the singleton primary or fallback AsyncGroq client."""
    global _client, _fallback_client
    if use_fallback:
        if _fallback_client is None:
            _fallback_client = AsyncGroq(
                api_key=settings.FALLBACK_GROQ_API_KEY,
                timeout=10.0,
            )
        return _fallback_client
    else:
        if _client is None:
            _client = AsyncGroq(
                api_key=settings.GROQ_API_KEY,
                timeout=10.0,
            )
        return _client


# ── Typed helpers ─────────────────────────────────────────────────────────────


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
        )
    return completion.choices[0].message.content or ""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def evaluate(messages: list[dict]) -> str:
    """Call the evaluation model (70B) with JSON output. Returns raw JSON string."""
    try:
        client = _get_client(use_fallback=False)
        completion = await client.chat.completions.create(
            model=settings.GROQ_EVAL_MODEL,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=settings.GROQ_EVAL_MAX_TOKENS,
            temperature=settings.GROQ_EVAL_TEMPERATURE,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.warning(
            "Groq evaluate failed with primary API key: %s. Retrying with fallback...",
            e,
        )
        client = _get_client(use_fallback=True)
        completion = await client.chat.completions.create(
            model=settings.GROQ_EVAL_MODEL,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=settings.GROQ_EVAL_MAX_TOKENS,
            temperature=settings.GROQ_EVAL_TEMPERATURE,
            response_format={"type": "json_object"},
        )
    return completion.choices[0].message.content or ""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def generate(messages: list[dict]) -> str:
    """Call the main generation model (70B) with JSON output. Returns raw JSON string."""
    try:
        client = _get_client(use_fallback=False)
        completion = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=settings.GROQ_MAX_TOKENS,
            temperature=settings.GROQ_TEMPERATURE,
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.warning(
            "Groq generate failed with primary API key: %s. Retrying with fallback...",
            e,
        )
        client = _get_client(use_fallback=True)
        completion = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=settings.GROQ_MAX_TOKENS,
            temperature=settings.GROQ_TEMPERATURE,
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
