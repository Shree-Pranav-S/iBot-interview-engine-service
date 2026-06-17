"""
AnswerEvaluation model.

Per-answer live evaluation records written immediately after each
candidate response during the interview — including silence and
irrelevant responses, not just substantive answers.

Drives the bot's next-action decision in real time and feeds the
holistic evaluation Celery task post-interview.

IMPORTANT: candidate_assessment_id is the sole reference key used for
all queries, including holistic evaluation loading. Do not query via
interview_session.id.
Owned by: interview-service
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Float, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.data.models.postgres.base import Base

if TYPE_CHECKING:
    from src.data.models.postgres.interview_session import InterviewSession


class AnswerEvaluation(Base):
    __tablename__ = "answer_evaluations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )

    # Sole cross-service reference key — used for holistic evaluation loading
    candidate_assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        index=True,
    )

    # FK to interview_sessions for ORM relationship only
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # References the candidate turn row this evaluation covers
    turn_number: Mapped[int] = mapped_column(Integer, nullable=False)

    # Section in which this answer occurred
    # e.g. "Java", "Spring Boot", "behavioural", "cultural"
    section: Mapped[str] = mapped_column(Text, nullable=False, index=True)

    # Skill under assessment — null for behavioural and cultural fit turns
    skill: Mapped[str | None] = mapped_column(Text, nullable=True)

    # The exact question the bot asked before this answer
    question_text: Mapped[str] = mapped_column(Text, nullable=False)

    # The candidate answer as transcribed by Deepgram Nova-3.
    # Empty string for silence turns.
    # Actual transcript of what was said for irrelevant turns.
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)

    # Classification of this response from the classify_response node.
    # answer | clarification | silence | irrelevant
    # Carried here so the holistic evaluation LLM has full classification
    # context when identifying patterns across the transcript.
    response_classification: Mapped[str] = mapped_column(Text, nullable=False)

    # strong | adequate | weak | non_answer | silence | irrelevant
    # silence and irrelevant are distinct states so the holistic evaluation
    # can differentiate a candidate who went silent from one who answered poorly.
    quality: Mapped[str] = mapped_column(Text, nullable=False)

    # 0.0 to 10.0
    # Set to 0.0 for silence after exhausted attempts and for all irrelevant responses
    score: Mapped[float] = mapped_column(Float, nullable=False)

    # Adaptive difficulty level when this question was asked: easy | medium | hard
    # Carried into holistic evaluation so the LLM contextualises each score —
    # a score of 7 on a hard question carries more weight than on an easy one.
    difficulty_at_time: Mapped[str] = mapped_column(
        Text, nullable=False, default="easy"
    )

    # Whether the bot asked the candidate to elaborate before this evaluation
    # was finalised. Carried into the holistic evaluation prompt so the LLM
    # knows the score reflects an answer given after prompting.
    nudge_given: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Signals the candidate demonstrated in their answer
    signals_present: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )

    # Signals the candidate did not demonstrate
    signals_missing: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )

    # SpeechBrain ECAPA-TDNN tone classification for this turn.
    # Format: {"confident": 0.6, "neutral": 0.3, "hesitant": 0.1}
    # Null for silence and irrelevant turns where audio is insufficient.
    tone_scores: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Brief one-line evaluator note carried into the holistic evaluation prompt
    one_line_feedback: Mapped[str | None] = mapped_column(Text, nullable=True)

    evaluated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    session: Mapped["InterviewSession"] = relationship(
        back_populates="answer_evaluations",
    )

    def __repr__(self) -> str:
        return (
            f"<AnswerEvaluation turn={self.turn_number} "
            f"skill={self.skill} score={self.score} quality={self.quality}>"
        )
