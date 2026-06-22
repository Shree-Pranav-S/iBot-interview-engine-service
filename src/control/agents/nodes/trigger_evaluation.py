"""Holistic evaluation trigger node."""

from __future__ import annotations

import logging

from src.control.agents.state import InterviewState
from src.core.services.evaluation_service import run_holistic_evaluation
from src.data.repositories import interview_workflow_repository as db

logger = logging.getLogger(__name__)


async def trigger_evaluation(state: InterviewState) -> dict:
    if state.get("holistic_evaluation_done"):
        return {}

    try:
        completed = await run_holistic_evaluation(
            str(state["candidate_assessment_id"]),
            state,  # type: ignore[arg-type]
        )
        if not completed:
            return {"holistic_evaluation_done": False, "next_node": "__end__"}
        await db.mark_evaluation_generated(state["candidate_assessment_id"])
    except Exception:
        logger.exception("Failed to run holistic interview evaluation")
        return {"holistic_evaluation_done": False, "next_node": "__end__"}

    final_status = (
        "TERMINATED" if state.get("session_status") == "TERMINATED" else "EVALUATED"
    )
    return {
        "holistic_evaluation_done": True,
        "session_status": final_status,
        "next_node": "__end__",
    }
