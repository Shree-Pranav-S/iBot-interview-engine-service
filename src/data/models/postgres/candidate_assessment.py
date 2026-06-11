"""CandidateAssessment model."""

import uuid
from datetime import datetime

from sqlalchemy import Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.data.models.postgres.base import Base
from src.data.models.postgres.mixins import TimestampMixin


class CandidateAssessment(TimestampMixin, Base):
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
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="INVITED", index=True
    )
    interview_started_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    interview_ended_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
