"""Database operations for durable event logs."""

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.models.postgres.event_log import EventLog
from src.schemas.event_log import EventLogCreate


class EventLogsRepository:
    """Persist immutable events and support retention-driven soft deletion."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, event: EventLogCreate) -> EventLog:
        record = EventLog(
            event_name=event.event_name.value,
            source_service=event.source_service.value,
            correlation_id=event.correlation_id,
            candidate_assessment_id=event.candidate_assessment_id,
            recruiter_id=event.recruiter_id,
            metadata_json=event.metadata,
            error_message=event.error_message,
            duration_ms=event.duration_ms,
        )
        self._session.add(record)
        await self._session.flush()
        return record

    async def create_many(
        self,
        events: Sequence[EventLogCreate],
    ) -> list[EventLog]:
        records = [
            EventLog(
                event_name=event.event_name.value,
                source_service=event.source_service.value,
                correlation_id=event.correlation_id,
                candidate_assessment_id=event.candidate_assessment_id,
                recruiter_id=event.recruiter_id,
                metadata_json=event.metadata,
                error_message=event.error_message,
                duration_ms=event.duration_ms,
            )
            for event in events
        ]
        if records:
            self._session.add_all(records)
            await self._session.flush()
        return records

    async def soft_delete(self, event_id: uuid.UUID) -> bool:
        result = await self._session.execute(
            update(EventLog)
            .where(EventLog.id == event_id)
            .where(EventLog.deleted_at.is_(None))
            .values(deleted_at=datetime.now(UTC))
        )
        return bool(result.rowcount)
