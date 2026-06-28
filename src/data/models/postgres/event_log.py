"""Immutable cross-service business and lifecycle event log."""

import uuid

from sqlalchemy import CheckConstraint, Index, Integer, Text, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from src.data.models.postgres.base import Base
from src.data.models.postgres.mixins import TimestampMixin


class EventLog(TimestampMixin, Base):
    """Durable event with soft deletion reserved for retention workflows."""

    __tablename__ = "event_logs"
    __table_args__ = (
        CheckConstraint(
            "duration_ms IS NULL OR duration_ms >= 0",
            name="ck_event_logs_duration_nonnegative",
        ),
        CheckConstraint(
            "jsonb_typeof(metadata) = 'object'",
            name="ck_event_logs_metadata_object",
        ),
        Index(
            "ix_event_logs_event_created_at",
            "event_name",
            "created_at",
        ),
        Index(
            "ix_event_logs_correlation_created_at",
            "correlation_id",
            "created_at",
        ),
        Index(
            "ix_event_logs_candidate_created_at",
            "candidate_assessment_id",
            "created_at",
            postgresql_where=text("candidate_assessment_id IS NOT NULL"),
        ),
        Index(
            "ix_event_logs_recruiter_created_at",
            "recruiter_id",
            "created_at",
            postgresql_where=text("recruiter_id IS NOT NULL"),
        ),
        Index("ix_event_logs_created_at", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )
    event_name: Mapped[str] = mapped_column(Text, nullable=False)
    source_service: Mapped[str] = mapped_column(Text, nullable=False)
    correlation_id: Mapped[str] = mapped_column(Text, nullable=False)
    candidate_assessment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    recruiter_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
    )
    metadata_json: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<EventLog id={self.id} event={self.event_name} "
            f"service={self.source_service}>"
        )
