"""Repository for assessment context consumed by the interview graph."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.data.models.postgres.assessment import Assessment
from src.data.models.postgres.candidate import Candidate
from src.data.models.postgres.candidate_assessment import CandidateAssessment
from src.data.models.postgres.recruiter import Recruiter


def _uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class AssessmentContextRepository:
    """Load and update assessment context in one caller-owned transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def load_interview_context(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> dict[str, Any]:
        """Load candidate, assessment, recruiter, and interview-plan context."""

        statement = (
            select(
                CandidateAssessment.interview_started_at,
                CandidateAssessment.resume_parsed,
                Candidate.full_name.label("candidate_name"),
                Assessment.interview_plan,
                Assessment.interview_duration_mins,
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
            .where(CandidateAssessment.id == _uuid(candidate_assessment_id))
        )
        result = await self._session.execute(statement)
        row = result.mappings().first()
        return dict(row) if row else {}

    async def mark_candidate_started(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> None:
        """Move a candidate assessment into the active interview state."""

        statement = (
            update(CandidateAssessment)
            .where(CandidateAssessment.id == _uuid(candidate_assessment_id))
            .values(status="IN_PROGRESS", updated_at=func.now())
        )
        await self._session.execute(statement)

    async def mark_candidate_timer_started(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> datetime:
        """Persist the first bot playout as the official interview start."""

        statement = (
            update(CandidateAssessment)
            .where(CandidateAssessment.id == _uuid(candidate_assessment_id))
            .values(
                status="IN_PROGRESS",
                interview_started_at=func.coalesce(
                    CandidateAssessment.interview_started_at,
                    func.now(),
                ),
                updated_at=func.now(),
            )
            .returning(CandidateAssessment.interview_started_at)
        )
        result = await self._session.execute(statement)
        started_at = result.scalar_one()
        if started_at is None:
            raise RuntimeError("Interview start timestamp was not persisted")
        return started_at

    async def mark_candidate_completed(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> None:
        """Persist the completed candidate lifecycle state."""

        statement = (
            update(CandidateAssessment)
            .where(CandidateAssessment.id == _uuid(candidate_assessment_id))
            .values(
                status="COMPLETED",
                interview_ended_at=func.coalesce(
                    CandidateAssessment.interview_ended_at,
                    func.now(),
                ),
                updated_at=func.now(),
            )
        )
        await self._session.execute(statement)
