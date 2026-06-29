"""Read/write mapping for assessment context owned by the core API."""

import uuid
from datetime import datetime

from sqlalchemy import Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.data.models.postgres.base import Base
from src.data.models.postgres.mixins import TimestampMixin


class Assessment(TimestampMixin, Base):
    """Assessment fields consumed by the interview engine."""

    __tablename__ = "assessments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    recruiter_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    role_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    jd_text: Mapped[str] = mapped_column(Text, nullable=False)
    jd_file_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    jd_analysis: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    focus_areas: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    interview_plan: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    interview_duration_mins: Mapped[int] = mapped_column(Integer, nullable=False)
    window_start: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
    )
    window_end: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(Text, nullable=False, index=True)
