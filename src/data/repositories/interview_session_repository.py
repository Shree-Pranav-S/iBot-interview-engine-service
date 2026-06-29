"""Repository for durable interview session state."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.models.postgres.interview_session import InterviewSession


def _uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _session_dict(session: InterviewSession) -> dict[str, Any]:
    return {
        column.key: getattr(session, column.key)
        for column in InterviewSession.__mapper__.column_attrs
    }


def _append_unique(
    items: list[dict[str, Any]] | None,
    item: dict[str, Any],
    *,
    id_key: str,
) -> tuple[list[dict[str, Any]], bool]:
    item_id = str(item.get(id_key) or "")
    if not item_id:
        raise ValueError(f"{id_key} is required for idempotent append")

    current = list(items or [])
    if any(str(existing.get(id_key) or "") == item_id for existing in current):
        return current, False
    return [*current, item], True


class InterviewSessionRepository:
    """Read and mutate interview state through one injected session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_session_by_candidate_assessment_id(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> dict[str, Any]:
        """Return the session belonging to a candidate assessment."""

        result = await self._session.execute(
            select(InterviewSession).where(
                InterviewSession.candidate_assessment_id
                == _uuid(candidate_assessment_id)
            )
        )
        session = result.scalar_one_or_none()
        return _session_dict(session) if session else {}

    async def get_or_create_session(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> dict[str, Any]:
        """Create the one durable session for a candidate when absent."""

        assessment_id = _uuid(candidate_assessment_id)
        statement = (
            insert(InterviewSession)
            .values(
                candidate_assessment_id=assessment_id,
                status="INITIALIZING",
                transcript=[],
                violations=[],
                total_elapsed_secs=0,
                total_pause_secs=0,
            )
            .on_conflict_do_nothing(
                index_elements=[InterviewSession.candidate_assessment_id],
            )
        )
        await self._session.execute(statement)
        result = await self._session.execute(
            select(InterviewSession)
            .where(
                InterviewSession.candidate_assessment_id == assessment_id,
            )
            .with_for_update()
        )
        return _session_dict(result.scalar_one())

    async def append_transcript_turn(
        self,
        session_id: str | uuid.UUID,
        turn: dict[str, Any],
        *,
        total_elapsed_secs: int | None = None,
    ) -> bool:
        """Append one transcript turn, idempotently keyed by ``turn_id``."""

        session = await self._locked_session(session_id)
        transcript, appended = _append_unique(
            session.transcript,
            turn,
            id_key="turn_id",
        )
        if appended:
            session.transcript = transcript
        if total_elapsed_secs is not None:
            session.total_elapsed_secs = max(
                int(session.total_elapsed_secs or 0),
                max(0, int(total_elapsed_secs)),
            )
        session.last_updated_at = func.now()
        await self._session.flush()
        return appended

    async def append_violation(
        self,
        session_id: str | uuid.UUID,
        violation: dict[str, Any],
    ) -> bool:
        """Append one violation, idempotently keyed by ``violation_id``."""

        session = await self._locked_session(session_id)
        violations, appended = _append_unique(
            session.violations,
            violation,
            id_key="violation_id",
        )
        if appended:
            session.violations = violations
            session.last_updated_at = func.now()
            await self._session.flush()
        return appended

    async def update_elapsed_time(
        self,
        session_id: str | uuid.UUID,
        *,
        elapsed_secs: int,
        total_pause_secs: int,
    ) -> None:
        """Persist monotonically increasing elapsed and pause totals."""

        statement = (
            update(InterviewSession)
            .where(InterviewSession.id == _uuid(session_id))
            .values(
                total_elapsed_secs=func.greatest(
                    InterviewSession.total_elapsed_secs,
                    max(0, int(elapsed_secs)),
                ),
                total_pause_secs=func.greatest(
                    InterviewSession.total_pause_secs,
                    max(0, int(total_pause_secs)),
                ),
                last_updated_at=func.now(),
            )
        )
        await self._session.execute(statement)

    async def mark_session_in_progress(
        self,
        session_id: str | uuid.UUID,
    ) -> None:
        """Mark a session active and clear its reconnect deadline."""

        statement = (
            update(InterviewSession)
            .where(InterviewSession.id == _uuid(session_id))
            .values(
                status="IN_PROGRESS",
                reconnect_deadline=None,
                last_updated_at=func.now(),
            )
        )
        await self._session.execute(statement)

    async def complete_session(
        self,
        session_id: str | uuid.UUID,
        *,
        total_elapsed_secs: int,
    ) -> None:
        """Mark a session completed before holistic evaluation."""

        statement = (
            update(InterviewSession)
            .where(InterviewSession.id == _uuid(session_id))
            .values(
                status="COMPLETED",
                total_elapsed_secs=max(0, int(total_elapsed_secs)),
                reconnect_deadline=None,
                active_connection_id=None,
                last_updated_at=func.now(),
            )
        )
        await self._session.execute(statement)

    async def _locked_session(
        self,
        session_id: str | uuid.UUID,
    ) -> InterviewSession:
        result = await self._session.execute(
            select(InterviewSession)
            .where(InterviewSession.id == _uuid(session_id))
            .with_for_update()
        )
        return result.scalar_one()
