"""Celery handlers for final holistic interview evaluation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.core.services.evaluation_service import run_holistic_evaluation
from src.data.clients.celery_client import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=2,
    name="interview.run_holistic_evaluation",
)
def run_holistic_evaluation_task(
    self: Any,
    candidate_assessment_id: str,
    state: dict[str, Any] | None = None,
) -> None:
    """Generate and persist the holistic evaluation report for one interview."""
    try:
        evaluation_saved = asyncio.run(
            run_holistic_evaluation(candidate_assessment_id, state or {})
        )
        if not evaluation_saved:
            raise RuntimeError("Holistic evaluation did not save a report")
    except Exception as exc:
        countdown = min(120, 10 * (self.request.retries + 1))
        logger.exception(
            "Holistic evaluation task failed",
            extra={
                "candidate_assessment_id": candidate_assessment_id,
                "retry_in": countdown,
            },
        )
        raise self.retry(exc=exc, countdown=countdown)


def enqueue_holistic_evaluation(
    candidate_assessment_id: str,
    state: dict[str, Any],
) -> str:
    """Queue final holistic evaluation on the interview evaluation worker."""
    result = run_holistic_evaluation_task.delay(candidate_assessment_id, state)
    return str(result.id)
