"""
TranscriptTurn model.

Append-only table storing every conversation turn.
One row per speaker turn written immediately after each turn completes.

Enables partial transcript evaluation on auto-submit and full transcript
replay in the recruiter report viewer.

IMPORTANT: candidate_assessment_id is used as the primary reference key
to stay consistent with the cross-service boundary design — do not join
via interview_session.id.
Owned by: interview-service
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.data.models.postgres.base import Base

if TYPE_CHECKING:
    from src.data.models.postgres.interview_session import InterviewSession


class TranscriptTurn(Base):
    __tablename__ = "transcript_turns"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )

    # Primary cross-service reference key.
    # Do not use interview_session FK — candidate_assessment_id is the
    # canonical reference across both services.
    candidate_assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    # FK to interview_sessions for ORM relationship only — not the
    # canonical cross-service key.
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Ordered sequence within the session, starting from 1
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # bot | candidate
    speaker: Mapped[str] = mapped_column(Text, nullable=False)

    # Transcribed candidate speech or generated bot text
    text: Mapped[str] = mapped_column(Text, nullable=False)

    # Section name active during this turn
    # e.g. "self_intro", "Java", "behavioural", "cultural"
    section: Mapped[str] = mapped_column(Text, nullable=False, index=True)

    # question | answer | clarification | silence | irrelevant |
    # acknowledgement | nudge | think_timer | transition | closing
    turn_type: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Classification assigned by the classify_response node.
    # answer | clarification | silence | irrelevant — null for bot turns.
    # Stored so the recruiter transcript viewer can render each candidate
    # turn with appropriate visual treatment and scoring context.
    response_classification: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Adaptive difficulty level when this question was asked.
    # easy | medium | hard — null for candidate turns.
    # Lets the recruiter understand the context behind each score.
    difficulty_at_time: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Deepgram Flux transcription confidence score — null for bot turns
    stt_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Concepts addressed in this turn, merged into LangGraph
    # used_concepts state after the row is written.
    concept_tags: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)

    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    session: Mapped["InterviewSession"] = relationship(
        back_populates="transcript_turns",
    )

    def __repr__(self) -> str:
        return (
            f"<TranscriptTurn turn={self.turn_number} "
            f"speaker={self.speaker} type={self.turn_type}>"
        )
