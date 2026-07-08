"""Non-blocking transcript and violation persistence nodes."""

from __future__ import annotations

from typing import Any

from src.control.agents.state import InterviewState
from src.control.agents.utils.persist_turn import (
    drain_background_persistence,
    schedule_bot_persistence,
    schedule_candidate_persistence,
    schedule_elapsed_persistence,
)

__all__ = [
    "drain_background_persistence",
    "persist_bot_output",
    "persist_candidate_output",
    "schedule_elapsed_persistence",
]


async def persist_bot_output(state: InterviewState) -> dict[str, Any]:
    """
    Graph node to queue the latest bot utterance for database persistence.

    Args:
        state: The current interview state.

    Returns:
        State updates clearing the pending bot turn.
    """

    schedule_bot_persistence(state)
    return {"pending_bot_turn": None}


async def persist_candidate_output(
    state: InterviewState,
) -> dict[str, Any]:
    """
    Graph node to queue candidate speech and violations for database persistence.
    Also increments the global turn counter.

    Args:
        state: The current interview state.

    Returns:
        State updates clearing the pending candidate turn/violations and incrementing turn number.
    """

    schedule_candidate_persistence(state)
    return {
        "pending_candidate_turn": None,
        "violations_to_persist": [],
        "turn_number": int(state.get("turn_number") or 1) + 1,
    }
