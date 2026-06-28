"""Strict JSON helpers for classification, evaluation, and generation."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from src.core.services import llm_service

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


async def classify_with_schema(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
) -> ResponseModelT:
    """Call the classifier and reject anything outside the Pydantic contract."""

    raw = await llm_service.classify(messages)
    return response_model.model_validate_json(raw)


async def evaluate_with_schema(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
) -> ResponseModelT:
    """Call the live evaluator and reject anything outside its contract."""

    raw = await llm_service.live_evaluate(messages)
    return response_model.model_validate_json(raw)


async def generate_with_schema(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
) -> ResponseModelT:
    """Call the question generator and reject non-schema JSON."""

    raw = await llm_service.generate(messages)
    return response_model.model_validate_json(raw)


async def rephrase_with_schema(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
) -> ResponseModelT:
    """Use the fast model for a short, strictly structured rephrase."""

    raw = await llm_service.lightweight(
        messages,
        max_tokens=128,
        temperature=0.1,
        json_mode=True,
    )
    return response_model.model_validate_json(raw)
