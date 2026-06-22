"""Incremental persistence node."""

from __future__ import annotations

from src.control.agents.state import InterviewState
from src.control.time_manager import compute_elapsed
from src.data.repositories import interview_workflow_repository as db


async def persist_turn(state: InterviewState) -> dict:
    state_for_db = dict(state)
    state_for_db["total_elapsed_secs"] = compute_elapsed(state)
    await db.persist_session_state(state_for_db)
    return {"total_elapsed_secs": state_for_db["total_elapsed_secs"]}
