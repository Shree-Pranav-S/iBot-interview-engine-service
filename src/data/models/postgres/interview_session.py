"""InterviewSession model."""

import uuid
from datetime import datetime

import sqlalchemy as sa
from sqlalchemy import Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from src.data.models.postgres.base import Base


class InterviewSession(Base):
    __tablename__ = "interview_sessions"
    __table_args__ = (
        UniqueConstraint(
            "candidate_assessment_id",
            name="uq_interview_sessions_candidate_assessment_id",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    candidate_assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="INITIALIZING",
        index=True,
    )
    current_section: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="self_intro",
    )
    section_progress: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default=sa.text("'{}'::jsonb"),
    )
    timer_started_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    total_elapsed_secs: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    total_pause_secs: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )
    paused_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    grace_period_expires_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    connectivity_events: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default=sa.text("'[]'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    last_updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )
