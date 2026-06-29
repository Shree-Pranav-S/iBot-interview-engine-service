"""Business service for durable event recording."""

import logging

from src.data.repositories.unit_of_work import InterviewUnitOfWork
from src.schemas.event_log import EventLogCreate

logger = logging.getLogger(__name__)


async def record_event_in_background(event: EventLogCreate) -> None:
    async with InterviewUnitOfWork() as unit_of_work:
        await unit_of_work.event_logs.create(event)


async def try_record_event_in_background(event: EventLogCreate) -> None:
    """Best-effort logging that never changes the interview or task outcome."""

    try:
        await record_event_in_background(event)
    except Exception:
        logger.exception(
            "Failed to persist event log",
            extra={
                "event_name": event.event_name.value,
                "correlation_id": event.correlation_id,
            },
        )
