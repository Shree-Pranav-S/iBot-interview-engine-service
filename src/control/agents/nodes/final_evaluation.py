"""LangGraph terminal node that queues holistic evaluation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.control.agents.state import InterviewState
from src.handlers.celery_tasks.evaluation_tasks import (
    enqueue_final_evaluation,
)

logger = logging.getLogger(__name__)


async def trigger_final_evaluation(
    state: InterviewState,
) -> dict[str, Any]:
    """
    Queue the idempotent background evaluation and let closing continue.

    This node acts as a bridge to the Celery evaluation task. It safely
    enqueues the holistic evaluation once the interview reaches a terminal state.

    Args:
        state: The current interview state.

    Returns:
        State updates containing the background task ID and status.
    """

    existing_task_id = str(state.get("holistic_evaluation_task_id") or "").strip()
    if existing_task_id:
        return {
            "holistic_evaluation_status": "QUEUED",
            "next_action": "end",
        }

    candidate_assessment_id = str(state["candidate_assessment_id"])
    try:
        task_id = await asyncio.to_thread(
            enqueue_final_evaluation,
            candidate_assessment_id,
        )
    except Exception:
        logger.exception(
            "Failed to enqueue final holistic evaluation",
            extra={
                "candidate_assessment_id": candidate_assessment_id,
            },
        )
        return {
            "holistic_evaluation_status": "ENQUEUE_FAILED",
            "next_action": "end",
        }

    logger.info(
        "Queued final holistic evaluation",
        extra={
            "candidate_assessment_id": candidate_assessment_id,
            "task_id": task_id,
        },
    )
    return {
        "holistic_evaluation_status": "QUEUED",
        "holistic_evaluation_task_id": task_id,
        "next_action": "end",
    }
