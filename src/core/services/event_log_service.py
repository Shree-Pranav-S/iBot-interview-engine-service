"""Business service for durable event recording."""

import logging

from src.data.clients.postgres_client import get_session_factory
from src.data.repositories.event_logs_repository import EventLogsRepository
from src.schemas.event_log import EventLogCreate

logger = logging.getLogger(__name__)


class EventLogService:
    def __init__(self, repository: EventLogsRepository) -> None:
        self._repository = repository

    async def record(self, event: EventLogCreate) -> None:
        await self._repository.create(event)


async def record_event_in_background(event: EventLogCreate) -> None:
    session_factory = await get_session_factory()
    async with session_factory() as session, session.begin():
        service = EventLogService(EventLogsRepository(session))
        await service.record(event)


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
