"""Atomic database operations for candidate session credentials and reconnects."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class CandidateSessionRepository:
    """Request/transaction-scoped access to invitation and session lifecycle rows."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def lock_invitation_context(
        self,
        invitation_token: uuid.UUID,
    ) -> dict[str, Any]:
        result = await self._session.execute(
            text(
                """
                SELECT
                    ca.id AS candidate_assessment_id,
                    ca.candidate_id,
                    ca.assessment_id,
                    ca.status AS candidate_assessment_status,
                    ca.invite_consumed,
                    ca.invite_consumed_at,
                    ca.interview_started_at,
                    c.full_name AS candidate_name,
                    a.title AS assessment_title,
                    a.interview_plan,
                    a.interview_duration_mins,
                    a.window_end,
                    COALESCE(r.company_name, '') AS company_name
                FROM candidate_assessments ca
                JOIN candidates c ON c.id = ca.candidate_id
                JOIN assessments a ON a.id = ca.assessment_id
                LEFT JOIN recruiters r ON r.id = a.recruiter_id
                WHERE ca.invitation_token = :invitation_token
                FOR UPDATE OF ca
                """
            ),
            {"invitation_token": invitation_token},
        )
        row = result.mappings().first()
        return dict(row) if row else {}

    async def get_or_create_session_for_update(
        self,
        candidate_assessment_id: uuid.UUID,
    ) -> dict[str, Any]:
        await self._session.execute(
            text(
                """
                INSERT INTO interview_sessions (
                    candidate_assessment_id,
                    status,
                    transcript,
                    violations,
                    total_elapsed_secs,
                    total_pause_secs,
                    disconnect_count,
                    timeout_disconnect_count
                )
                VALUES (
                    :candidate_assessment_id,
                    'INITIALIZING',
                    '[]'::jsonb,
                    '[]'::jsonb,
                    0,
                    0,
                    0,
                    0
                )
                ON CONFLICT (candidate_assessment_id) DO NOTHING
                """
            ),
            {"candidate_assessment_id": candidate_assessment_id},
        )
        result = await self._session.execute(
            text(
                """
                SELECT
                    id,
                    candidate_assessment_id,
                    status,
                    session_token,
                    session_token_expires_at,
                    disconnect_count,
                    timeout_disconnect_count,
                    last_disconnected_at,
                    reconnect_deadline,
                    active_connection_id,
                    total_elapsed_secs,
                    total_pause_secs
                FROM interview_sessions
                WHERE candidate_assessment_id = :candidate_assessment_id
                FOR UPDATE
                """
            ),
            {"candidate_assessment_id": candidate_assessment_id},
        )
        row = result.mappings().one()
        return dict(row)

    async def consume_invitation(
        self,
        candidate_assessment_id: uuid.UUID,
    ) -> None:
        await self._session.execute(
            text(
                """
                UPDATE candidate_assessments
                SET invite_consumed = TRUE,
                    invite_consumed_at = COALESCE(invite_consumed_at, NOW()),
                    updated_at = NOW()
                WHERE id = :candidate_assessment_id
                  AND invite_consumed = FALSE
                """
            ),
            {"candidate_assessment_id": candidate_assessment_id},
        )

    async def set_session_credentials(
        self,
        session_id: uuid.UUID,
        *,
        session_token: str,
        expires_at: datetime,
    ) -> None:
        await self._session.execute(
            text(
                """
                UPDATE interview_sessions
                SET session_token = :session_token,
                    session_token_expires_at = :expires_at,
                    last_updated_at = NOW()
                WHERE id = :session_id
                """
            ),
            {
                "session_id": session_id,
                "session_token": session_token,
                "expires_at": expires_at,
            },
        )

    async def lock_session_context_by_token(
        self,
        session_token: str,
    ) -> dict[str, Any]:
        result = await self._session.execute(
            text(
                """
                SELECT
                    s.id AS session_id,
                    s.candidate_assessment_id,
                    s.status AS session_status,
                    s.session_token,
                    s.session_token_expires_at,
                    s.disconnect_count,
                    s.timeout_disconnect_count,
                    s.last_disconnected_at,
                    s.reconnect_deadline,
                    s.active_connection_id,
                    s.total_elapsed_secs,
                    s.total_pause_secs,
                    ca.candidate_id,
                    ca.assessment_id,
                    ca.status AS candidate_assessment_status,
                    ca.interview_started_at,
                    c.full_name AS candidate_name,
                    a.title AS assessment_title,
                    a.interview_plan,
                    a.interview_duration_mins,
                    a.window_end,
                    COALESCE(r.company_name, '') AS company_name
                FROM interview_sessions s
                JOIN candidate_assessments ca
                    ON ca.id = s.candidate_assessment_id
                JOIN candidates c ON c.id = ca.candidate_id
                JOIN assessments a ON a.id = ca.assessment_id
                LEFT JOIN recruiters r ON r.id = a.recruiter_id
                WHERE s.session_token = :session_token
                FOR UPDATE OF s, ca
                """
            ),
            {"session_token": session_token},
        )
        row = result.mappings().first()
        return dict(row) if row else {}

    async def bind_active_connection(
        self,
        session_id: uuid.UUID,
        *,
        connection_id: str,
    ) -> None:
        await self._session.execute(
            text(
                """
                UPDATE interview_sessions
                SET active_connection_id = :connection_id,
                    last_updated_at = NOW()
                WHERE id = :session_id
                  AND status IN ('INITIALIZING', 'IN_PROGRESS')
                """
            ),
            {
                "session_id": session_id,
                "connection_id": connection_id,
            },
        )

    async def restore_disconnected_session(
        self,
        session_id: uuid.UUID,
    ) -> dict[str, Any]:
        result = await self._session.execute(
            text(
                """
                UPDATE interview_sessions
                SET status = 'IN_PROGRESS',
                    total_pause_secs = total_pause_secs + COALESCE(
                        GREATEST(
                            0,
                            EXTRACT(
                                EPOCH FROM (NOW() - last_disconnected_at)
                            )::integer
                        ),
                        0
                    ),
                    reconnect_deadline = NULL,
                    active_connection_id = NULL,
                    last_updated_at = NOW()
                WHERE id = :session_id
                  AND status = 'DISCONNECTED'
                RETURNING *
                """
            ),
            {"session_id": session_id},
        )
        row = result.mappings().one()
        await self._session.execute(
            text(
                """
                UPDATE candidate_assessments
                SET status = 'IN_PROGRESS',
                    updated_at = NOW()
                WHERE id = :candidate_assessment_id
                """
            ),
            {"candidate_assessment_id": row["candidate_assessment_id"]},
        )
        return dict(row)

    async def terminate_reconnect_timeout(
        self,
        session_id: uuid.UUID,
    ) -> dict[str, Any]:
        result = await self._session.execute(
            text(
                """
                UPDATE interview_sessions
                SET status = 'TERMINATED',
                    timeout_disconnect_count = timeout_disconnect_count + 1,
                    reconnect_deadline = NULL,
                    active_connection_id = NULL,
                    last_updated_at = NOW()
                WHERE id = :session_id
                  AND status = 'DISCONNECTED'
                RETURNING *
                """
            ),
            {"session_id": session_id},
        )
        row = result.mappings().one()
        await self._mark_candidate_terminated(row["candidate_assessment_id"])
        return dict(row)

    async def record_disconnect(
        self,
        session_id: uuid.UUID,
        *,
        connection_id: str,
        reconnect_deadline: datetime,
        elapsed_secs: int,
    ) -> dict[str, Any]:
        result = await self._session.execute(
            text(
                """
                UPDATE interview_sessions
                SET disconnect_count = disconnect_count + 1,
                    status = CASE
                        WHEN disconnect_count + 1 >= 3
                        THEN 'TERMINATED'
                        ELSE 'DISCONNECTED'
                    END,
                    total_elapsed_secs = GREATEST(
                        total_elapsed_secs,
                        :elapsed_secs
                    ),
                    last_disconnected_at = NOW(),
                    reconnect_deadline = CASE
                        WHEN disconnect_count + 1 >= 3
                        THEN NULL
                        ELSE :reconnect_deadline
                    END,
                    active_connection_id = NULL,
                    last_updated_at = NOW()
                WHERE id = :session_id
                  AND active_connection_id = :connection_id
                  AND status IN ('INITIALIZING', 'IN_PROGRESS')
                RETURNING *
                """
            ),
            {
                "session_id": session_id,
                "connection_id": connection_id,
                "reconnect_deadline": reconnect_deadline,
                "elapsed_secs": max(0, int(elapsed_secs)),
            },
        )
        row = result.mappings().first()
        if not row:
            return {}
        outcome = dict(row)
        if outcome["status"] == "TERMINATED":
            await self._mark_candidate_terminated(outcome["candidate_assessment_id"])
        return outcome

    async def _mark_candidate_terminated(
        self,
        candidate_assessment_id: uuid.UUID,
    ) -> None:
        await self._session.execute(
            text(
                """
                UPDATE candidate_assessments
                SET status = 'TERMINATED',
                    interview_ended_at = COALESCE(interview_ended_at, NOW()),
                    updated_at = NOW()
                WHERE id = :candidate_assessment_id
                """
            ),
            {"candidate_assessment_id": candidate_assessment_id},
        )
