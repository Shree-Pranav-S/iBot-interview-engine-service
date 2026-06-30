"""Typed Structured Outputs helpers for interview workflow LLM calls."""

from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from src.core.services import llm_service

ResponseModelT = TypeVar("ResponseModelT", bound=BaseModel)


async def classify_with_schema(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
    *,
    key_slot: int | None = None,
) -> ResponseModelT:
    """
    Call the classifier with its API-enforced Pydantic contract.

    Args:
        messages: The chat history to send to the LLM.
        response_model: The Pydantic model defining the response structure.

    Returns:
        A validated instance of the response model.

    Raises:
        ValidationError: If semantic Pydantic validators reject the response.
    """

    return await llm_service.classify(
        messages,
        response_model,
        key_slot=key_slot,
    )


async def evaluate_with_schema(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
    *,
    key_slot: int | None = None,
) -> ResponseModelT:
    """
    Call the live evaluator LLM and reject anything outside its contract.

    Args:
        messages: The chat history to send to the LLM.
        response_model: The Pydantic model defining the response structure.

    Returns:
        A validated instance of the response model.
    """

    return await llm_service.live_evaluate(
        messages,
        response_model,
        key_slot=key_slot,
    )


async def generate_with_schema(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
    *,
    key_slot: int | None = None,
) -> ResponseModelT:
    """
    Call the question generator with its API-enforced Pydantic contract.

    Args:
        messages: The chat history to send to the LLM.
        response_model: The Pydantic model defining the response structure.

    Returns:
        A validated instance of the response model.
    """

    return await llm_service.generate(
        messages,
        response_model,
        key_slot=key_slot,
    )


async def interviewer_turn_with_schema(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
    *,
    key_slot: int | None = None,
) -> ResponseModelT:
    """
    Call the merged interviewer-turn model with its API-enforced contract.

    Args:
        messages: The chat history to send to the LLM.
        response_model: The Pydantic model defining the response structure.

    Returns:
        A validated instance of the response model.
    """

    return await llm_service.respond(
        messages,
        response_model,
        key_slot=key_slot,
    )


async def rephrase_with_schema(
    messages: list[dict[str, str]],
    response_model: type[ResponseModelT],
    *,
    key_slot: int | None = None,
) -> ResponseModelT:
    """
    Use the lightweight/fast model for a short, strictly structured rephrase.

    Args:
        messages: The chat history to send to the LLM.
        response_model: The Pydantic model defining the response structure.

    Returns:
        A validated instance of the response model.
    """

    return await llm_service.lightweight(
        messages,
        response_model,
        key_slot=key_slot,
        max_tokens=128,
        temperature=0.1,
    )
