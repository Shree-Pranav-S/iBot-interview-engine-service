"""Shared candidate-assessment lifecycle mapping."""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from src.data.models.postgres.base import Base
from src.data.models.postgres.mixins import TimestampMixin


class CandidateAssessment(TimestampMixin, Base):
    """Candidate interview lifecycle fields used by this service."""

    __tablename__ = "candidate_assessments"
    __table_args__ = (
        UniqueConstraint(
            "candidate_id",
            "assessment_id",
            name="uq_candidate_assessment",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    resume_file_path: Mapped[str] = mapped_column(Text, nullable=False)
    resume_parsed: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    resume_parse_status: Mapped[str] = mapped_column(Text, nullable=False)
    invitation_token: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        unique=True,
    )
    invite_consumed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )
    invite_consumed_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="INVITED", index=True
    )
    interview_started_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    interview_ended_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    recruiter_decision: Mapped[str] = mapped_column(Text, nullable=False)
    recruiter_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)
    reminders_sent: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )
