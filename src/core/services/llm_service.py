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

_clients: dict[tuple[str, str], AsyncGroq] = {}
_key_unavailable_until: dict[str, float] = {}
_disabled_keys: dict[str, str] = {}
ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


def _clean_api_key(value: str | None) -> str:
    cleaned = str(value or "").strip().strip('"').strip("'")
    if cleaned.startswith("${") and cleaned.endswith("}"):
        reference = cleaned[2:-1].strip()
        return _clean_api_key(getattr(settings, reference, ""))
    return cleaned


def _api_key_pool(purpose: str) -> list[str]:
    """Return the deduplicated common live-interview key pool."""

    configured_pool = [
        _clean_api_key(settings.GROQ_INTERVIEW_API_KEY_1),
        _clean_api_key(settings.GROQ_INTERVIEW_API_KEY_2),
        _clean_api_key(settings.GROQ_INTERVIEW_API_KEY_3),
        _clean_api_key(settings.GROQ_INTERVIEW_API_KEY_4),
    ]
    source_pool = [
        _clean_api_key(settings.GROQ_API_KEY),
        _clean_api_key(settings.FALLBACK_GROQ_API_KEY),
        _clean_api_key(settings.GROQ_QUESTION_API_KEY),
        _clean_api_key(settings.FALLBACK_GROQ_EVALUATION_KEY),
    ]
    candidates = [key for key in configured_pool if key] or [
        key for key in source_pool if key
    ]
    pool = list(dict.fromkeys(candidates))
    if not pool:
        raise RuntimeError(f"Groq {purpose} API key is not configured.")
    return pool


def _get_groq_client(purpose: str, api_key: str) -> AsyncGroq:
    """Return a lazily created client for one purpose/key combination."""

    cache_key = (purpose, api_key)
    if cache_key not in _clients:
        timeouts = {
            "classification": settings.GROQ_CLASSIFY_TIMEOUT_SECS,
            "question": settings.GROQ_QUESTION_TIMEOUT_SECS,
            "interviewer": settings.GROQ_INTERVIEWER_TIMEOUT_SECS,
        }
        _clients[cache_key] = AsyncGroq(
            api_key=api_key,
            timeout=timeouts[purpose],
            max_retries=0,
        )
    return _clients[cache_key]


def _error_code(exc: Exception) -> str:
    """Extract a stable provider error code without logging response secrets."""

    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    if isinstance(error, dict):
        return str(error.get("code") or "").strip()
    return str(body.get("code") or "").strip()


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _failure_policy(exc: Exception) -> tuple[bool, float | None, str]:
    """Return whether to rotate, the cooldown, and a non-secret reason."""

    status = getattr(exc, "status_code", None)
    code = _error_code(exc)
    permanent_codes = {
        "organization_restricted",
        "blocked_api_access",
        "invalid_api_key",
    }
    if code in permanent_codes or status == 401:
        return True, None, code or "authentication_failed"
    if status == 429:
        cooldown = _retry_after_seconds(exc)
        return (
            True,
            cooldown
            if cooldown is not None
            else settings.GROQ_KEY_RATE_LIMIT_COOLDOWN_SECS,
            code or "rate_limited",
        )
    if status in {403, 408, 409, 425} or (isinstance(status, int) and status >= 500):
        return True, settings.GROQ_KEY_TRANSIENT_COOLDOWN_SECS, code or f"http_{status}"
    if status is None:
        return (
            True,
            settings.GROQ_KEY_TRANSIENT_COOLDOWN_SECS,
            code or "transport_error",
        )
    # A normal 400 generally means the model/request/schema is invalid. Trying
    # the same request with other credentials only wastes quota and latency.
    return False, 0.0, code or f"http_{status}"


def _ordered_candidates(
    purpose: str,
    key_slot: int | None,
) -> tuple[list[tuple[int, str]], int]:
    """Rotate the pool by turn affinity and omit disabled/cooling-down keys."""

    pool = _api_key_pool(purpose)
    start = int(key_slot or 0) % len(pool)
    ordered = [
        ((start + offset) % len(pool), pool[(start + offset) % len(pool)])
        for offset in range(len(pool))
    ]
    now = time.monotonic()
    candidates = [
        (index, key)
        for index, key in ordered
        if key not in _disabled_keys and _key_unavailable_until.get(key, 0.0) <= now
    ]
    if not candidates:
        raise RuntimeError(
            f"All configured Groq keys for {purpose} are disabled or cooling down."
        )
    return candidates, len(pool)


def _mark_failed_key(
    key: str,
    *,
    cooldown_secs: float | None,
    reason: str,
) -> None:
    if cooldown_secs is None:
        _disabled_keys[key] = reason
        _key_unavailable_until.pop(key, None)
        return
    _key_unavailable_until[key] = time.monotonic() + max(0.0, cooldown_secs)


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


def _usage_log_fields(completion: Any) -> dict[str, Any]:
    """Extract token usage for structured logging."""

    usage = getattr(completion, "usage", None)
    if usage is None:
        return {}
    fields: dict[str, Any] = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
    cached_tokens = getattr(usage, "cached_tokens", None)
    details = getattr(usage, "prompt_tokens_details", None)
    if cached_tokens is None and details is not None:
        cached_tokens = getattr(details, "cached_tokens", None)
        if cached_tokens is None and isinstance(details, dict):
            cached_tokens = details.get("cached_tokens")
    fields["cached_tokens"] = cached_tokens
    return {key: value for key, value in fields.items() if value is not None}


async def _call_with_fallback(
    purpose: str,
    response_model: type[ResponseModelT],
    *,
    key_slot: int | None = None,
    **kwargs: Any,
) -> ResponseModelT:
    """Execute one strict-schema call through the turn-affine healthy-key pool."""

    started_at = time.perf_counter()
    kwargs["reasoning_effort"] = settings.GROQ_REASONING_EFFORT
    kwargs["response_format"] = _structured_response_format(response_model)
    candidates, pool_size = _ordered_candidates(purpose, key_slot)
    last_exc: Exception | None = None

    for attempt, (key_index, api_key) in enumerate(candidates, start=1):
        try:
            client = _get_groq_client(purpose, api_key)
            completion = await client.chat.completions.create(**kwargs)
        except Exception as exc:
            should_rotate, cooldown_secs, reason = _failure_policy(exc)
            if not should_rotate:
                raise
            last_exc = exc
            _mark_failed_key(
                api_key,
                cooldown_secs=cooldown_secs,
                reason=reason,
            )
            logger.warning(
                "Groq call failed; trying the next healthy interview key",
                extra={
                    "purpose": purpose,
                    "key_index": key_index,
                    "pool_size": pool_size,
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error_code": _error_code(exc) or None,
                    "status_code": getattr(exc, "status_code", None),
                    "cooldown_secs": cooldown_secs,
                    "elapsed_ms": round(
                        (time.perf_counter() - started_at) * 1000,
                        2,
                    ),
                },
            )
            continue

        _key_unavailable_until.pop(api_key, None)
        logger.info(
            "Groq call completed",
            extra={
                "purpose": purpose,
                "key_index": key_index,
                "pool_size": pool_size,
                "attempt": attempt,
                "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
                **_usage_log_fields(completion),
            },
        )
        raw_content = completion.choices[0].message.content or ""
        return response_model.model_validate_json(raw_content)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"No healthy Groq key was available for {purpose}.")


async def generate(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
    *,
    key_slot: int | None = None,
) -> ResponseModelT:
    """Generate one question through strict Structured Outputs."""
    return await _call_with_fallback(
        purpose="question",
        response_model=response_model,
        key_slot=key_slot,
        model=settings.GROQ_QUESTION_MODEL,
        messages=messages,
        max_tokens=settings.GROQ_QUESTION_MAX_TOKENS,
        temperature=settings.GROQ_QUESTION_TEMPERATURE,
    )


async def respond(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
    *,
    key_slot: int | None = None,
) -> ResponseModelT:
    """Run the merged classify/evaluate/generate interviewer-turn call.

    This is the single in-interview reasoning call: it routes the candidate
    utterance, judges the answer, and produces the next spoken question or
    clarification in one round trip.
    """
    return await _call_with_fallback(
        purpose="interviewer",
        response_model=response_model,
        key_slot=key_slot,
        model=settings.GROQ_INTERVIEWER_MODEL,
        messages=messages,
        max_tokens=settings.GROQ_INTERVIEWER_MAX_TOKENS,
        temperature=settings.GROQ_INTERVIEWER_TEMPERATURE,
    )


async def lightweight(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
    *,
    key_slot: int | None = None,
    max_tokens: int = 192,
    temperature: float = 0.3,
) -> ResponseModelT:
    """Run a short classifier-model completion through strict Structured Outputs."""
    return await _call_with_fallback(
        purpose="classification",
        response_model=response_model,
        key_slot=key_slot,
        model=settings.GROQ_CLASSIFY_MODEL,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
