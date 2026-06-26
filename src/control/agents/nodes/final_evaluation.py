"""Final evaluation trigger node for completed interviews."""

from __future__ import annotations

import logging
from typing import Any

from src.control.agents.state import InterviewState
from src.core.services.evaluation_service import run_holistic_evaluation
from src.data.repositories import (
    assessment_context_repository,
    interview_session_repository,
)
from src.handlers.celery_tasks.evaluation_tasks import enqueue_holistic_evaluation

logger = logging.getLogger(__name__)


async def final_evaluation(state: InterviewState) -> dict[str, Any]:
    """Complete the session and trigger the holistic evaluation process."""
    session_id = str(state["interview_session_id"])
    candidate_assessment_id = str(state["candidate_assessment_id"])
    elapsed_secs = int(state.get("elapsed_secs") or 0)
    evaluation_state = {**state, "interview_session_id": session_id}

    await interview_session_repository.complete_session(
        session_id,
        total_elapsed_secs=elapsed_secs,
    )
    await assessment_context_repository.mark_candidate_completed(
        candidate_assessment_id
    )

    final_evaluation_status = "queued"
    holistic_evaluation_task_id = None
    try:
        holistic_evaluation_task_id = enqueue_holistic_evaluation(
            candidate_assessment_id,
            evaluation_state,
        )
    except Exception:
        logger.exception(
            "Failed to queue holistic evaluation",
            extra={"candidate_assessment_id": candidate_assessment_id},
        )
        final_evaluation_status = (
            "completed_inline"
            if await run_holistic_evaluation(candidate_assessment_id, evaluation_state)
            else "failed"
        )

    return {
        "session_status": "COMPLETED",
        "closing_done": True,
        "should_close": True,
        "next_action": "complete",
        "final_evaluation_status": final_evaluation_status,
        "holistic_evaluation_task_id": holistic_evaluation_task_id,
    }


trigger_final_evaluation_placeholder = final_evaluation
