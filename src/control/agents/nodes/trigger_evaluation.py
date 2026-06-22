"""Holistic evaluation trigger node."""

from __future__ import annotations

import logging

from src.control.agents.state import InterviewState
from src.data.repositories import interview_workflow_repository as db

logger = logging.getLogger(__name__)


async def trigger_evaluation(state: InterviewState) -> dict:
    if state.get("holistic_evaluation_done"):
        return {}
    try:
        await db.upsert_interview_evaluation(state)  # type: ignore
    except Exception:
        logger.exception("Failed to write holistic interview evaluation")
    return {"holistic_evaluation_done": True, "next_node": "__end__"}
