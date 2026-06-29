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
    """
    Call the classifier LLM and reject anything outside the Pydantic contract.

    Args:
        messages: The chat history to send to the LLM.
        response_model: The Pydantic model defining the expected JSON structure.

    Returns:
        A validated instance of the response model.

    Raises:
        ValidationError: If the LLM output does not match the schema.
    """

    raw = await llm_service.classify(messages)
    return response_model.model_validate_json(raw)


async def evaluate_with_schema(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
) -> ResponseModelT:
    """
    Call the live evaluator LLM and reject anything outside its contract.

    Args:
        messages: The chat history to send to the LLM.
        response_model: The Pydantic model defining the expected JSON structure.

    Returns:
        A validated instance of the response model.
    """

    raw = await llm_service.live_evaluate(messages)
    return response_model.model_validate_json(raw)


async def generate_with_schema(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
) -> ResponseModelT:
    """
    Call the question generator LLM and reject non-schema JSON.

    Args:
        messages: The chat history to send to the LLM.
        response_model: The Pydantic model defining the expected JSON structure.

    Returns:
        A validated instance of the response model.
    """

    raw = await llm_service.generate(messages)
    return response_model.model_validate_json(raw)


async def rephrase_with_schema(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
) -> ResponseModelT:
    """
    Use the lightweight/fast model for a short, strictly structured rephrase.

    Args:
        messages: The chat history to send to the LLM.
        response_model: The Pydantic model defining the expected JSON structure.

    Returns:
        A validated instance of the response model.
    """

    raw = await llm_service.lightweight(
        messages,
        max_tokens=128,
        temperature=0.1,
        json_mode=True,
    )
    return response_model.model_validate_json(raw)
