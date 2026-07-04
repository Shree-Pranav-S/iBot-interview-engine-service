"""Celery entrypoint for completed-interview holistic evaluation."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from src.core.services.evaluation_errors import (
    PermanentEvaluationError,
    TransientEvaluationError,
)
from src.core.services.evaluation_service import (
    mark_holistic_evaluation_failed,
    run_holistic_evaluation,
)
from src.core.services.event_log_service import try_record_event_in_background
from src.data.clients.celery_client import celery_app
from src.schemas.event_log import EventLogCreate, EventName, EventSource

logger = logging.getLogger(__name__)
_event_loop: asyncio.AbstractEventLoop | None = None


def _run_async(coroutine: Any) -> Any:
    """
    Reuse one loop so pooled async clients never cross closed event loops.

    Args:
        coroutine: The asynchronous coroutine to execute.

    Returns:
        The result of the executed coroutine.
    """

    global _event_loop
    if _event_loop is None or _event_loop.is_closed():
        _event_loop = asyncio.new_event_loop()
    return _event_loop.run_until_complete(coroutine)


def _persist_failure(candidate_assessment_id: str) -> None:
    """
    Mark an evaluation as permanently failed in the database.

    Args:
        candidate_assessment_id: The UUID string of the candidate assessment.
    """
    try:
        _run_async(mark_holistic_evaluation_failed(candidate_assessment_id))
    except Exception:
        logger.exception(
            "Could not persist holistic evaluation failure status",
            extra={
                "candidate_assessment_id": candidate_assessment_id,
            },
        )


def _record_task_event(
    event_name: EventName,
    *,
    task: Any,
    task_id: str,
    candidate_assessment_id: str,
    started_at: float | None = None,
    error: BaseException | None = None,
    **metadata: Any,
) -> None:
    """Persist one consistently shaped Celery lifecycle event."""

    event_metadata = {
        "task_name": task.name,
        "retry_number": task.request.retries,
        **metadata,
    }
    duration_ms = (
        int((time.monotonic() - started_at) * 1000) if started_at is not None else None
    )
    _run_async(
        try_record_event_in_background(
            EventLogCreate(
                event_name=event_name,
                source_service=EventSource.INTERVIEW_ENGINE,
                correlation_id=task_id,
                candidate_assessment_id=uuid.UUID(candidate_assessment_id),
                metadata=event_metadata,
                error_message=str(error) if error is not None else None,
                duration_ms=duration_ms,
            )
        )
    )


def _record_terminal_failure(
    *,
    task: Any,
    task_id: str,
    candidate_assessment_id: str,
    started_at: float,
    error: BaseException,
    log_message: str,
) -> None:
    """Persist terminal failure state, emit its event, and log the exception."""

    _persist_failure(candidate_assessment_id)
    _record_task_event(
        EventName.CELERY_TASK_FAILED,
        task=task,
        task_id=task_id,
        candidate_assessment_id=candidate_assessment_id,
        started_at=started_at,
        error=error,
        exception_type=type(error).__name__,
    )
    logger.exception(
        log_message,
        extra={"candidate_assessment_id": candidate_assessment_id},
    )


@celery_app.task(
    bind=True,
    max_retries=2,
    name="core.process_final_evaluation",
)
def process_final_evaluation_task(
    self: Any,
    candidate_assessment_id: str,
) -> dict[str, Any]:
    """
    Evaluate one finalized session with bounded transient retries.

    Args:
        self: The Celery task instance.
        candidate_assessment_id: The UUID string of the candidate assessment.

    Returns:
        A dictionary containing the evaluation result or status.

    Raises:
        PermanentEvaluationError: If a fundamental issue prevents evaluation.
        TransientEvaluationError: If a temporary rate limit or connection issue occurs.
    """

    task_id = str(self.request.id or candidate_assessment_id)
    started_at = time.monotonic()
    _record_task_event(
        EventName.CELERY_TASK_STARTED,
        task=self,
        task_id=task_id,
        candidate_assessment_id=candidate_assessment_id,
    )
    try:
        result = _run_async(run_holistic_evaluation(candidate_assessment_id))
        _record_task_event(
            EventName.CELERY_TASK_COMPLETED,
            task=self,
            task_id=task_id,
            candidate_assessment_id=candidate_assessment_id,
            started_at=started_at,
            result_status=result.get("status"),
        )
        return result
    except PermanentEvaluationError as exc:
        _record_terminal_failure(
            task=self,
            task_id=task_id,
            candidate_assessment_id=candidate_assessment_id,
            started_at=started_at,
            error=exc,
            log_message="Holistic evaluation failed permanently; not retrying",
        )
        raise
    except Exception as exc:
        if not isinstance(exc, TransientEvaluationError):
            _record_terminal_failure(
                task=self,
                task_id=task_id,
                candidate_assessment_id=candidate_assessment_id,
                started_at=started_at,
                error=exc,
                log_message="Unexpected non-retryable holistic evaluation failure",
            )
            raise

        if self.request.retries >= int(self.max_retries or 0):
            _record_terminal_failure(
                task=self,
                task_id=task_id,
                candidate_assessment_id=candidate_assessment_id,
                started_at=started_at,
                error=exc,
                log_message="Holistic evaluation exhausted transient retries",
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
        _record_task_event(
            EventName.CELERY_TASK_RETRYING,
            task=self,
            task_id=task_id,
            candidate_assessment_id=candidate_assessment_id,
            started_at=started_at,
            error=exc,
            retry_in_seconds=countdown,
            exception_type=type(exc).__name__,
        )
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
    """
    Queue evaluation after closing playout has time to finalize the session.

    Args:
        candidate_assessment_id: The UUID string of the candidate assessment.

    Returns:
        The celery task ID string for the enqueued evaluation.
    """

    result = process_final_evaluation_task.apply_async(
        args=[candidate_assessment_id],
        countdown=10,
    )
    return str(result.id)
