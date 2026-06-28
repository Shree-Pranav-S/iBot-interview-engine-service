"""Repository for loading assessment context used by the interview graph."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import text

from src.data.clients.postgres_client import get_session_factory


def _uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


async def load_interview_context(
    candidate_assessment_id: str | uuid.UUID,
) -> dict[str, Any]:
    """Load candidate, candidate_assessment, and assessment context."""

    session_factory = await get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """
                SELECT
                    ca.id AS candidate_assessment_id,
                    ca.candidate_id,
                    ca.assessment_id,
                    ca.status AS candidate_assessment_status,
                    ca.interview_started_at,
                    ca.resume_parsed,
                    c.full_name AS candidate_name,
                    c.email AS candidate_email,
                    a.title AS assessment_title,
                    a.role_name,
                    a.jd_text,
                    a.jd_analysis,
                    a.focus_areas,
                    a.interview_plan,
                    a.interview_duration_mins,
                    COALESCE(r.company_name, '') AS company_name
                FROM candidate_assessments ca
                JOIN candidates c ON ca.candidate_id = c.id
                JOIN assessments a ON ca.assessment_id = a.id
                LEFT JOIN recruiters r ON a.recruiter_id = r.id
                WHERE ca.id = :candidate_assessment_id
                """
            ),
            {"candidate_assessment_id": _uuid(candidate_assessment_id)},
        )
        row = result.mappings().first()
        return dict(row) if row else {}


async def mark_candidate_started(candidate_assessment_id: str | uuid.UUID) -> None:
    """Move candidate_assessments into IN_PROGRESS while audio is preparing."""

    session_factory = await get_session_factory()
    async with session_factory() as session:
        await session.execute(
            text(
                """
                UPDATE candidate_assessments
                SET status = 'IN_PROGRESS',
                    updated_at = NOW()
                WHERE id = :candidate_assessment_id
                """
            ),
            {"candidate_assessment_id": _uuid(candidate_assessment_id)},
        )
        await session.commit()


async def mark_candidate_timer_started(
    candidate_assessment_id: str | uuid.UUID,
) -> datetime:
    """Record the first bot playout as the interview's official start."""

    session_factory = await get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """
                UPDATE candidate_assessments
                SET status = 'IN_PROGRESS',
                    interview_started_at = COALESCE(interview_started_at, NOW()),
                    updated_at = NOW()
                WHERE id = :candidate_assessment_id
                RETURNING interview_started_at
                """
            ),
            {"candidate_assessment_id": _uuid(candidate_assessment_id)},
        )
        started_at = result.scalar_one()
        await session.commit()
        return started_at


async def mark_candidate_completed(candidate_assessment_id: str | uuid.UUID) -> None:
    """Move candidate_assessments into COMPLETED after graph placeholder end."""

    session_factory = await get_session_factory()
    async with session_factory() as session:
        await session.execute(
            text(
                """
                UPDATE candidate_assessments
                SET status = 'COMPLETED',
                    interview_ended_at = COALESCE(interview_ended_at, NOW()),
                    updated_at = NOW()
                WHERE id = :candidate_assessment_id
                """
            ),
            {"candidate_assessment_id": _uuid(candidate_assessment_id)},
        )
        await session.commit()
