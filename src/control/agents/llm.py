"""Groq LLM helpers used by interview graph nodes."""

from __future__ import annotations

import json
import logging

from groq import AsyncGroq

from src.config.settings import settings

logger = logging.getLogger(__name__)

_client: AsyncGroq | None = None
_fallback_client: AsyncGroq | None = None


def _get_client(use_fallback: bool = False) -> AsyncGroq:
    global _client, _fallback_client
    if use_fallback:
        if _fallback_client is None:
            _fallback_client = AsyncGroq(
                api_key=settings.FALLBACK_GROQ_API_KEY, timeout=15.0
            )
        return _fallback_client
    else:
        if _client is None:
            _client = AsyncGroq(api_key=settings.GROQ_API_KEY, timeout=15.0)
        return _client


async def groq_complete(
    *,
    system: str,
    user: str,
    max_tokens: int = 500,
    temperature: float = 0.3,
    json_mode: bool = False,
    model: str | None = None,
) -> str:
    """Call Groq chat completions with a consistent interface."""
    kwargs: dict = {
        "model": model or settings.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        if not settings.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not configured.")
        client = _get_client(use_fallback=False)
        completion = await client.chat.completions.create(**kwargs)
    except Exception as e:
        logger.warning(
            "Groq complete failed with primary API key: %s. Retrying with fallback...",
            e,
        )
        client = _get_client(use_fallback=True)
        completion = await client.chat.completions.create(**kwargs)

    return completion.choices[0].message.content or ""


def parse_json(text: str) -> dict:
    """Parse JSON returned by an LLM, tolerating markdown fences."""
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        clean = "\n".join(lines[1:-1]) if len(lines) > 2 else clean
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON: %r", clean[:300])
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def groq_eot_classify(text: str, question: str) -> str:
    """Return complete or incomplete for end-of-turn detection."""
    try:
        result = await groq_complete(
            system=(
                "You are an end-of-turn classifier for a voice interview. "
                "Reply with exactly one word: complete or incomplete."
            ),
            user=(
                f"Interview question: {question}\n"
                f"Candidate response so far: {text}\n"
                "Is this response complete, or is the speaker mid-thought?"
            ),
            max_tokens=5,
            temperature=0.0,
            model=settings.GROQ_CLASSIFY_MODEL,
        )
    except Exception:
        logger.exception("Groq EOT classifier failed; defaulting to complete")
        return "complete"

    lowered = result.strip().lower()
    return "incomplete" if "incomplete" in lowered else "complete"
