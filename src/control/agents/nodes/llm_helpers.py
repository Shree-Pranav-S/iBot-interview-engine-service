"""LLM helpers used by LangGraph interview nodes."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.config.settings import settings
from src.core.services import llm_service

logger = logging.getLogger(__name__)
PromptResponseT = TypeVar("PromptResponseT", bound=BaseModel)


def compact_json(value: Any, *, max_chars: int = 2400) -> str:
    """Serialize context compactly and bound it for prompt use."""

    try:
        text = json.dumps(value, ensure_ascii=True, default=str)
    except TypeError:
        text = json.dumps(str(value), ensure_ascii=True)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def clean_text(value: Any, *, max_chars: int = 420) -> str:
    """Normalize generated speech text for concise TTS output."""

    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return truncated.rstrip(".,;:") + "."


def _extract_json(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        loaded = json.loads(raw)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        return {}
    try:
        loaded = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _has_llm_key() -> bool:
    if os.getenv("IBOT_DISABLE_LLM_CALLS") == "1":
        return False
    primary = str(settings.GROQ_API_KEY or "").strip().strip('"').strip("'")
    fallback = str(settings.FALLBACK_GROQ_API_KEY or "").strip().strip('"').strip("'")
    return bool(primary or fallback)


def _extra_error_locs(exc: ValidationError) -> list[str]:
    locs: list[str] = []
    for error in exc.errors():
        if error.get("type") != "extra_forbidden":
            return []
        loc = ".".join(str(part) for part in error.get("loc", ()))
        locs.append(loc)
    return locs


def _validate_model_response(
    *,
    label: str,
    parsed: dict[str, Any],
    response_model: type[PromptResponseT],
) -> PromptResponseT:
    try:
        return response_model.model_validate(parsed)
    except ValidationError as exc:
        extra_locs = _extra_error_locs(exc)
        if not extra_locs:
            raise

        logger.warning(
            "LLM %s response included extra keys; ignoring them",
            label,
            extra={
                "response_model": response_model.__name__,
                "extra_keys": extra_locs,
            },
        )
        allowed_fields = set(response_model.model_fields)
        filtered = {
            key: value for key, value in parsed.items() if key in allowed_fields
        }
        return response_model.model_validate(filtered)


async def _call_json(
    label: str,
    caller: Callable[[list[dict[str, str]]], Awaitable[str]],
    messages: list[dict[str, str]],
    response_model: type[PromptResponseT],
) -> PromptResponseT:
    """Call a JSON-mode LLM helper and validate the response schema."""

    if not _has_llm_key():
        raise RuntimeError(f"LLM {label} call requires a Groq API key")

    try:
        raw = await caller(messages)
        parsed = _extract_json(raw)
        if not parsed:
            raise ValueError(f"LLM {label} returned no JSON object")
        return _validate_model_response(
            label=label,
            parsed=parsed,
            response_model=response_model,
        )
    except ValidationError:
        logger.exception("LLM %s response failed schema validation", label)
        raise
    except Exception:
        logger.exception("LLM %s call failed", label)
        raise


def model_to_dict(value: BaseModel | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    return dict(value)


async def classify_json(
    messages: list[dict[str, str]],
    response_model: type[PromptResponseT],
) -> PromptResponseT:
    async def caller(inner_messages: list[dict[str, str]]) -> str:
        return await llm_service.lightweight(
            inner_messages,
            max_tokens=256,
            temperature=0.0,
            json_mode=True,
        )

    return await _call_json("classification", caller, messages, response_model)


async def generate_json(
    messages: list[dict[str, str]],
    response_model: type[PromptResponseT],
) -> PromptResponseT:
    return await _call_json(
        "generation", llm_service.generate, messages, response_model
    )


async def evaluate_json(
    messages: list[dict[str, str]],
    response_model: type[PromptResponseT],
) -> PromptResponseT:
    return await _call_json(
        "evaluation", llm_service.evaluate, messages, response_model
    )


async def live_evaluate_json(
    messages: list[dict[str, str]],
    response_model: type[PromptResponseT],
) -> PromptResponseT:
    return await _call_json(
        "live_evaluation", llm_service.live_evaluate, messages, response_model
    )


async def lightweight_json(
    messages: list[dict[str, str]],
    response_model: type[PromptResponseT],
    *,
    max_tokens: int = 220,
    temperature: float = 0.4,
) -> PromptResponseT:
    async def caller(inner_messages: list[dict[str, str]]) -> str:
        return await llm_service.lightweight(
            inner_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=True,
        )

    return await _call_json("lightweight", caller, messages, response_model)
