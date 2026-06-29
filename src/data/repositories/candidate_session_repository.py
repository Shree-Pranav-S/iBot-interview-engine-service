"""Atomic repository operations for candidate entry and reconnection."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import Integer, case, cast, func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.models.postgres.assessment import Assessment
from src.data.models.postgres.candidate import Candidate
from src.data.models.postgres.candidate_assessment import CandidateAssessment
from src.data.models.postgres.interview_session import InterviewSession
from src.data.models.postgres.recruiter import Recruiter


def _session_dict(session: InterviewSession) -> dict[str, Any]:
    return {
        column.key: getattr(session, column.key)
        for column in InterviewSession.__mapper__.column_attrs
    }


class CandidateSessionRepository:
    """Access invitation and reconnect state through one injected session."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def lock_invitation_context(
        self,
        invitation_token: uuid.UUID,
    ) -> dict[str, Any]:
        """Lock and return the invitation lifecycle context."""

        statement = (
            select(
                CandidateAssessment.id.label("candidate_assessment_id"),
                CandidateAssessment.candidate_id,
                CandidateAssessment.assessment_id,
                CandidateAssessment.status.label("candidate_assessment_status"),
                CandidateAssessment.invite_consumed,
                CandidateAssessment.invite_consumed_at,
                CandidateAssessment.interview_started_at,
                Candidate.full_name.label("candidate_name"),
                Assessment.title.label("assessment_title"),
                Assessment.interview_plan,
                Assessment.interview_duration_mins,
                Assessment.window_end,
                func.coalesce(Recruiter.company_name, "").label("company_name"),
            )
            .join(
                Candidate,
                Candidate.id == CandidateAssessment.candidate_id,
            )
            .join(
                Assessment,
                Assessment.id == CandidateAssessment.assessment_id,
            )
            .outerjoin(Recruiter, Recruiter.id == Assessment.recruiter_id)
            .where(CandidateAssessment.invitation_token == invitation_token)
            .with_for_update(of=CandidateAssessment)
        )
        result = await self._session.execute(statement)
        row = result.mappings().first()
        return dict(row) if row else {}

    async def get_or_create_session_for_update(
        self,
        candidate_assessment_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Create the durable session when absent, then lock and return it."""

        statement = (
            insert(InterviewSession)
            .values(
                candidate_assessment_id=candidate_assessment_id,
                status="INITIALIZING",
                transcript=[],
                violations=[],
                total_elapsed_secs=0,
                total_pause_secs=0,
                disconnect_count=0,
                timeout_disconnect_count=0,
            )
            .on_conflict_do_nothing(
                index_elements=[InterviewSession.candidate_assessment_id],
            )
        )
        await self._session.execute(statement)

        result = await self._session.execute(
            select(InterviewSession)
            .where(InterviewSession.candidate_assessment_id == candidate_assessment_id)
            .with_for_update()
        )
        return _session_dict(result.scalar_one())

    async def consume_invitation(
        self,
        candidate_assessment_id: uuid.UUID,
    ) -> None:
        """Mark an invitation consumed exactly once."""

        statement = (
            update(CandidateAssessment)
            .where(CandidateAssessment.id == candidate_assessment_id)
            .where(CandidateAssessment.invite_consumed.is_(False))
            .values(
                invite_consumed=True,
                invite_consumed_at=func.coalesce(
                    CandidateAssessment.invite_consumed_at,
                    func.now(),
                ),
                updated_at=func.now(),
            )
        )
        await self._session.execute(statement)

    async def set_session_credentials(
        self,
        session_id: uuid.UUID,
        *,
        session_token: str,
        expires_at: datetime,
    ) -> None:
        """Persist the browser session credential and expiry."""

        statement = (
            update(InterviewSession)
            .where(InterviewSession.id == session_id)
            .values(
                session_token=session_token,
                session_token_expires_at=expires_at,
                last_updated_at=func.now(),
            )
        )
        await self._session.execute(statement)

    async def lock_session_context_by_token(
        self,
        session_token: str,
    ) -> dict[str, Any]:
        """Lock and return the session and candidate context for a token."""

        statement = (
            select(
                InterviewSession.id.label("session_id"),
                InterviewSession.candidate_assessment_id,
                InterviewSession.status.label("session_status"),
                InterviewSession.session_token,
                InterviewSession.session_token_expires_at,
                InterviewSession.disconnect_count,
                InterviewSession.timeout_disconnect_count,
                InterviewSession.last_disconnected_at,
                InterviewSession.reconnect_deadline,
                InterviewSession.active_connection_id,
                InterviewSession.total_elapsed_secs,
                InterviewSession.total_pause_secs,
                CandidateAssessment.candidate_id,
                CandidateAssessment.assessment_id,
                CandidateAssessment.status.label("candidate_assessment_status"),
                CandidateAssessment.interview_started_at,
                Candidate.full_name.label("candidate_name"),
                Assessment.title.label("assessment_title"),
                Assessment.interview_plan,
                Assessment.interview_duration_mins,
                Assessment.window_end,
                func.coalesce(Recruiter.company_name, "").label("company_name"),
            )
            .join(
                CandidateAssessment,
                CandidateAssessment.id == InterviewSession.candidate_assessment_id,
            )
            .join(
                Candidate,
                Candidate.id == CandidateAssessment.candidate_id,
            )
            .join(
                Assessment,
                Assessment.id == CandidateAssessment.assessment_id,
            )
            .outerjoin(Recruiter, Recruiter.id == Assessment.recruiter_id)
            .where(InterviewSession.session_token == session_token)
            .with_for_update(of=[InterviewSession, CandidateAssessment])
        )
        result = await self._session.execute(statement)
        row = result.mappings().first()
        return dict(row) if row else {}

    async def bind_active_connection(
        self,
        session_id: uuid.UUID,
        *,
        connection_id: str,
    ) -> None:
        """Bind the current LiveKit job generation to an active session."""

        statement = (
            update(InterviewSession)
            .where(InterviewSession.id == session_id)
            .where(InterviewSession.status.in_(("INITIALIZING", "IN_PROGRESS")))
            .values(
                active_connection_id=connection_id,
                last_updated_at=func.now(),
            )
        )
        await self._session.execute(statement)

    async def restore_disconnected_session(
        self,
        session_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Restore a disconnected session and account for its paused time."""

        pause_seconds = func.coalesce(
            func.greatest(
                0,
                cast(
                    func.extract(
                        "epoch",
                        func.now() - InterviewSession.last_disconnected_at,
                    ),
                    Integer,
                ),
            ),
            0,
        )
        statement = (
            update(InterviewSession)
            .where(InterviewSession.id == session_id)
            .where(InterviewSession.status == "DISCONNECTED")
            .values(
                status="IN_PROGRESS",
                total_pause_secs=InterviewSession.total_pause_secs + pause_seconds,
                reconnect_deadline=None,
                active_connection_id=None,
                last_updated_at=func.now(),
            )
            .returning(InterviewSession)
        )
        result = await self._session.execute(statement)
        session = result.scalar_one()
        await self._session.execute(
            update(CandidateAssessment)
            .where(
                CandidateAssessment.id == session.candidate_assessment_id,
            )
            .values(status="IN_PROGRESS", updated_at=func.now())
        )
        return _session_dict(session)

    async def terminate_reconnect_timeout(
        self,
        session_id: uuid.UUID,
    ) -> dict[str, Any]:
        """Terminate a session whose reconnection grace period expired."""

        statement = (
            update(InterviewSession)
            .where(InterviewSession.id == session_id)
            .where(InterviewSession.status == "DISCONNECTED")
            .values(
                status="TERMINATED",
                timeout_disconnect_count=(
                    InterviewSession.timeout_disconnect_count + 1
                ),
                reconnect_deadline=None,
                active_connection_id=None,
                last_updated_at=func.now(),
            )
            .returning(InterviewSession)
        )
        result = await self._session.execute(statement)
        session = result.scalar_one()
        await self._mark_candidate_terminated(session.candidate_assessment_id)
        return _session_dict(session)

    async def record_disconnect(
        self,
        session_id: uuid.UUID,
        *,
        connection_id: str,
        reconnect_deadline: datetime,
        elapsed_secs: int,
    ) -> dict[str, Any]:
        """Record one current connection drop and ignore stale generations."""

        next_disconnect_count = InterviewSession.disconnect_count + 1
        should_terminate = next_disconnect_count >= 3
        statement = (
            update(InterviewSession)
            .where(InterviewSession.id == session_id)
            .where(InterviewSession.active_connection_id == connection_id)
            .where(InterviewSession.status.in_(("INITIALIZING", "IN_PROGRESS")))
            .values(
                disconnect_count=next_disconnect_count,
                status=case(
                    (should_terminate, "TERMINATED"),
                    else_="DISCONNECTED",
                ),
                total_elapsed_secs=func.greatest(
                    InterviewSession.total_elapsed_secs,
                    max(0, int(elapsed_secs)),
                ),
                last_disconnected_at=func.now(),
                reconnect_deadline=case(
                    (should_terminate, None),
                    else_=reconnect_deadline,
                ),
                active_connection_id=None,
                last_updated_at=func.now(),
            )
            .returning(InterviewSession)
        )
        result = await self._session.execute(statement)
        session = result.scalar_one_or_none()
        if session is None:
            return {}

        outcome = _session_dict(session)
        if session.status == "TERMINATED":
            await self._mark_candidate_terminated(
                session.candidate_assessment_id,
            )
        return outcome

    async def _mark_candidate_terminated(
        self,
        candidate_assessment_id: uuid.UUID,
    ) -> None:
        statement = (
            update(CandidateAssessment)
            .where(CandidateAssessment.id == candidate_assessment_id)
            .values(
                status="TERMINATED",
                interview_ended_at=func.coalesce(
                    CandidateAssessment.interview_ended_at,
                    func.now(),
                ),
                updated_at=func.now(),
            )
        )
        await self._session.execute(statement)
