"""Repository helpers for LiveKit interview room access."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from src.data.clients.postgres_client import get_session_factory


def _uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


async def _fetch_one(query: str, params: dict[str, Any]) -> dict[str, Any]:
    session_factory = await get_session_factory()
    async with session_factory() as session:
        result = await session.execute(text(query), params)
        row = result.mappings().first()
        return dict(row) if row else {}


async def _execute(query: str, params: dict[str, Any]) -> None:
    session_factory = await get_session_factory()
    async with session_factory() as session:
        await session.execute(text(query), params)
        await session.commit()


async def _load_candidate_context(candidate_assessment_id: str) -> dict[str, Any]:
    return await _fetch_one(
        """
        SELECT
            ca.id AS candidate_assessment_id,
            ca.status AS candidate_assessment_status,
            ca.resume_parsed,
            a.title AS assessment_title,
            a.role_name,
            a.interview_plan,
            a.interview_duration_mins,
            COALESCE(r.company_name, '') AS company_name
        FROM candidate_assessments ca
        JOIN assessments a ON ca.assessment_id = a.id
        LEFT JOIN recruiters r ON a.recruiter_id = r.id
        WHERE ca.id = :candidate_assessment_id
        """,
        {"candidate_assessment_id": _uuid(candidate_assessment_id)},
    )


async def _get_existing_session(candidate_assessment_id: str) -> dict[str, Any]:
    return await _fetch_one(
        """
        SELECT id, status, grace_period_expires_at
        FROM interview_sessions
        WHERE candidate_assessment_id = :candidate_assessment_id
        """,
        {"candidate_assessment_id": _uuid(candidate_assessment_id)},
    )


async def _mark_session_deactivated(candidate_assessment_id: str) -> None:
    await _execute(
        """
        UPDATE interview_sessions
        SET status = 'DEACTIVATED',
            last_updated_at = NOW()
        WHERE candidate_assessment_id = :candidate_assessment_id
        """,
        {"candidate_assessment_id": _uuid(candidate_assessment_id)},
    )


async def assert_session_can_start(candidate_assessment_id: str) -> tuple[bool, str]:
    """Return whether a candidate can receive LiveKit room credentials."""

    context = await _load_candidate_context(candidate_assessment_id)
    if not context:
        return False, "Candidate assessment was not found."

    ca_status = str(context.get("candidate_assessment_status") or "")
    if ca_status in {"COMPLETED", "EVALUATED"}:
        return False, "This interview has already been completed."

    session = await _get_existing_session(candidate_assessment_id)
    if not session:
        return True, ""

    session_status = str(session.get("status") or "")
    if session_status in {"COMPLETED", "EVALUATED", "TERMINATED", "DEACTIVATED"}:
        return False, "This interview session is already closed."

    expires_at = session.get("grace_period_expires_at")
    if (
        session_status == "PAUSED"
        and expires_at is not None
        and expires_at < datetime.now(UTC)
    ):
        await _mark_session_deactivated(candidate_assessment_id)
        return False, "The reconnection grace period has expired."

    return True, ""


async def load_candidate_livekit_context(
    invitation_token: str | uuid.UUID,
) -> dict[str, Any]:
    """Resolve a candidate invitation token for LiveKit room access."""

    return await _fetch_one(
        """
        SELECT
            ca.id AS candidate_assessment_id,
            ca.candidate_id,
            ca.assessment_id,
            ca.status AS candidate_assessment_status,
            COALESCE(c.full_name, '') AS candidate_name
        FROM candidate_assessments ca
        JOIN candidates c ON ca.candidate_id = c.id
        WHERE ca.invitation_token = :invitation_token
        """,
        {"invitation_token": _uuid(invitation_token)},
    )
