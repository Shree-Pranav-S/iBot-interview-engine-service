"""Celery entrypoint for completed-interview holistic evaluation."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy.exc import DBAPIError, OperationalError

from src.core.services.evaluation_errors import (
    PermanentEvaluationError,
    TransientEvaluationError,
)
from src.core.services.evaluation_service import run_holistic_evaluation
from src.data.clients.celery_client import celery_app
from src.data.repositories.evaluation_repository import (
    mark_evaluation_failed,
)

logger = logging.getLogger(__name__)
RETRYABLE_SQL_STATES = {"40001", "40P01"}
_event_loop: asyncio.AbstractEventLoop | None = None


def _run_async(coroutine: Any) -> Any:
    """Reuse one loop so pooled async clients never cross closed event loops."""

    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
    return _event_loop.run_until_complete(coroutine)


def _persist_failure(candidate_assessment_id: str) -> None:
    try:
        _run_async(mark_evaluation_failed(candidate_assessment_id))
    except Exception:
        logger.exception(
            "Could not persist holistic evaluation failure status",
            extra={
                "candidate_assessment_id": candidate_assessment_id,
            },
        )


def _retryable_database_error(exc: BaseException) -> bool:
    if isinstance(exc, OperationalError):
        return True
    if not isinstance(exc, DBAPIError):
        return False
    original = getattr(exc, "orig", None)
    sql_state = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
    return str(sql_state or "") in RETRYABLE_SQL_STATES


@celery_app.task(
    bind=True,
    max_retries=2,
    name="core.process_final_evaluation",
)
def process_final_evaluation_task(
    self: Any,
    candidate_assessment_id: str,
) -> dict[str, Any]:
    """Evaluate one finalized session with bounded transient retries."""

    try:
        return _run_async(run_holistic_evaluation(candidate_assessment_id))
    except PermanentEvaluationError:
        _persist_failure(candidate_assessment_id)
        logger.exception(
            "Holistic evaluation failed permanently; not retrying",
            extra={
                "candidate_assessment_id": candidate_assessment_id,
            },
        )
        raise
    except Exception as exc:
        if not (
            isinstance(exc, TransientEvaluationError) or _retryable_database_error(exc)
        ):
            _persist_failure(candidate_assessment_id)
            logger.exception(
                "Unexpected non-retryable holistic evaluation failure",
                extra={
                    "candidate_assessment_id": candidate_assessment_id,
                },
            )
            raise

        if self.request.retries >= int(self.max_retries or 0):
            _persist_failure(candidate_assessment_id)
            logger.exception(
                "Holistic evaluation exhausted transient retries",
                extra={
                    "candidate_assessment_id": candidate_assessment_id,
                },
            )
            raise

        exponential_delay = min(
            300,
            30 * (2**self.request.retries),
        )
        provider_delay = (
            exc.retry_after_seconds
            if isinstance(exc, TransientEvaluationError)
            else None
        )
        countdown = max(exponential_delay, provider_delay or 0)
        logger.warning(
            "Retrying transient holistic evaluation failure",
            exc_info=True,
            extra={
                "candidate_assessment_id": candidate_assessment_id,
                "retry_in_seconds": countdown,
                "retry_number": self.request.retries + 1,
            },
        )
        raise self.retry(exc=exc, countdown=countdown)


def enqueue_final_evaluation(
    candidate_assessment_id: str,
) -> str:
    """Queue evaluation after closing playout has time to finalize the session."""

    result = process_final_evaluation_task.apply_async(
        args=[candidate_assessment_id],
        countdown=10,
    )
    return str(result.id)
