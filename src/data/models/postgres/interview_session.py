"""
InterviewSession model.

Durable session state table for the interview engine. LangGraph checkpoints
store graph execution state separately; this table owns transcript, violations,
elapsed time, and lifecycle status for one candidate_assessment_id.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Integer, Text
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from src.data.models.postgres.base import Base


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )

    candidate_assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        unique=True,
        index=True,
    )

    transcript: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
    )

    violations: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        server_default="[]",
    )

    status: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="INITIALIZING",
        index=True,
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

    session_token: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        unique=True,
    )
    session_token_expires_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )
    disconnect_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    timeout_disconnect_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    last_disconnected_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )
    reconnect_deadline: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
    )
    active_connection_id: Mapped[str | None] = mapped_column(Text, nullable=True)

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

    def __repr__(self) -> str:
        return f"<InterviewSession id={self.id} status={self.status}>"
