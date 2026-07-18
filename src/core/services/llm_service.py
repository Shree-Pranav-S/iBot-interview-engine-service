"""Pooled Groq clients for latency-sensitive, in-interview LLM calls.

Final holistic evaluation is isolated in ``evaluation_llm_client`` and uses
NVIDIA NIM instead of this module.
"""

from __future__ import annotations

import logging
from typing import cast

from src.config.settings import settings
from src.utils.llm_service_helpers import ResponseModelT, _call_with_fallback

logger = logging.getLogger(__name__)


async def generate(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
    *,
    key_slot: int | None = None,
) -> ResponseModelT:
    """Generate one question through strict Structured Outputs."""
    return cast(
        ResponseModelT,
        await _call_with_fallback(
            purpose="question",
            response_model=response_model,
            key_slot=key_slot,
            model=settings.GROQ_QUESTION_MODEL,
            messages=messages,
            max_tokens=settings.GROQ_QUESTION_MAX_TOKENS,
            temperature=settings.GROQ_QUESTION_TEMPERATURE,
        ),
    )


async def respond(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
    *,
    key_slot: int | None = None,
) -> ResponseModelT:
    """Run stage-two evaluation and interviewer response generation."""
    return cast(
        ResponseModelT,
        await _call_with_fallback(
            purpose="interviewer",
            response_model=response_model,
            key_slot=key_slot,
            model=settings.GROQ_INTERVIEWER_MODEL,
            messages=messages,
            max_tokens=settings.GROQ_INTERVIEWER_MAX_TOKENS,
            temperature=settings.GROQ_INTERVIEWER_TEMPERATURE,
        ),
    )


async def classify(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
    *,
    key_slot: int | None = None,
) -> ResponseModelT:
    """Classify one candidate utterance with the dedicated fast model."""

    return cast(
        ResponseModelT,
        await _call_with_fallback(
            purpose="classification",
            response_model=response_model,
            key_slot=key_slot,
            model=settings.GROQ_CLASSIFY_MODEL,
            messages=messages,
            max_tokens=settings.GROQ_CLASSIFY_MAX_TOKENS,
            temperature=settings.GROQ_CLASSIFY_TEMPERATURE,
        ),
    )


async def lightweight(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
    *,
    key_slot: int | None = None,
    max_tokens: int = 192,
    temperature: float = 0.3,
) -> ResponseModelT:
    """Run a fast, general-purpose text completion (e.g. rephrasing, formatting) using the smaller/faster model."""
    return cast(
        ResponseModelT,
        await _call_with_fallback(
            purpose="classification",
            response_model=response_model,
            key_slot=key_slot,
            model=settings.GROQ_CLASSIFY_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ),
    )
