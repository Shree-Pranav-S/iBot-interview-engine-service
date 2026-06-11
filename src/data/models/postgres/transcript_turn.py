"""TranscriptTurn model."""

import uuid
from datetime import datetime

from sqlalchemy import Integer, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import ARRAY, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from src.data.models.postgres.base import Base


class TranscriptTurn(Base):
    __tablename__ = "transcript_turns"
    __table_args__ = (
        UniqueConstraint(
            "candidate_assessment_id",
            "turn_number",
            name="uq_transcript_turns_candidate_assessment_turn_number",
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
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)
    speaker: Mapped[str] = mapped_column(Text, nullable=False)
    text_content: Mapped[str] = mapped_column("text", Text, nullable=False)
    section: Mapped[str] = mapped_column(Text, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    stt_confidence: Mapped[float | None] = mapped_column(nullable=True)
    turn_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    concept_tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
