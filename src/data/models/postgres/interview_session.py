"""
InterviewSession model.

Core session state table for the interview-service.
Created when the candidate clicks Start Interview.
Updated at key lifecycle moments by the application layer.

LangGraph's native AsyncPostgresSaver manages its own checkpoint tables
(checkpoints, checkpoint_blobs, checkpoint_writes) independently.
Session recovery uses:
    graph.ainvoke(None, config={"configurable": {"thread_id": candidate_assessment_id}})
which loads the LangGraph checkpoint automatically.

IMPORTANT: candidate_assessment_id is the primary cross-service reference key
that joins back to all core-api tables.
Owned by: interview-service
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.data.models.postgres.base import Base

if TYPE_CHECKING:
    from src.data.models.postgres.answer_evaluation import AnswerEvaluation
    from src.data.models.postgres.interview_evaluation import InterviewEvaluation
    from src.data.models.postgres.transcript_turn import TranscriptTurn


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        server_default=func.gen_random_uuid(),
    )

    # One session per candidate per assessment — cross-service reference key
    candidate_assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        unique=True,
        index=True,
    )

    # INITIALIZING | IN_PROGRESS | PAUSED | COMPLETED | EVALUATED | DEACTIVATED | TERMINATED
    status: Mapped[str] = mapped_column(
        Text, nullable=False, default="INITIALIZING", index=True
    )

    # ── Section tracking ──────────────────────────────────────────────────────

    # Human-readable section name updated on every section transition
    # e.g. "self_intro", "Java", "Spring Boot", "behavioural", "cultural"
    current_section: Mapped[str] = mapped_column(
        Text, nullable=False, default="self_intro"
    )

    # Zero-based index into the ordered sections array in the interview plan.
    # Used by the LangGraph section transition node to advance without
    # re-parsing the plan on every turn.
    current_section_index: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    # Per-section runtime object keyed by section name.
    # Each entry: {time_budget_secs, time_elapsed_secs, questions_asked,
    #              avg_score, difficulty_reached, is_complete}
    # Populated progressively as sections complete.
    section_progress: Mapped[dict] = mapped_column(
        JSONB, nullable=False, default=dict, server_default="{}"
    )

    # ── Current turn state ────────────────────────────────────────────────────

    # Exact text of the most recently delivered bot question.
    # Persisted so reconnection can re-deliver the last question
    # without re-generating it via LLM.
    current_question_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Adaptive difficulty within the active section: easy | medium | hard
    # Never escalates above medium for skills with priority score <= 4.
    current_difficulty: Mapped[str] = mapped_column(
        Text, nullable=False, default="easy"
    )

    # Count of consecutive strong-quality answers in the current section.
    # Reset to 0 on section transition. Drives difficulty escalation.
    consecutive_strong_answers: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    # Count of consecutive weak or non-answer quality answers in current section.
    # Reset to 0 on section transition. Drives difficulty recalibration downward.
    consecutive_weak_answers: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    # Concept tags addressed across all turns in the current section.
    # Injected into every question generation prompt to prevent repetition.
    # Reset on section transition.
    used_concepts: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default="{}"
    )

    # ── Per-question sub-state ────────────────────────────────────────────────

    # Stage of the silence handling sub-flow for the current question.
    # 0 = no silence yet, 1 = first nudge delivered, 2 = think timer offered.
    # Reset to 0 on each new question.
    silence_attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Whether the bot has already asked the candidate to elaborate on the
    # current question. Prevents infinite elaboration loops.
    # Reset to FALSE on each new question.
    nudge_given_this_turn: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # ── Violation tracking ────────────────────────────────────────────────────

    # Running count of irrelevant responses across the entire interview.
    # At 3 the terminate node fires and status is set to TERMINATED.
    irrelevant_strike_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    # Ordered log of all session violations.
    # Each entry: {turn_number, violation_type, transcript, timestamp}
    # Surfaced in the recruiter report to provide context for low scores.
    violations: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )

    # ── Timer fields ──────────────────────────────────────────────────────────

    # Set when candidate clicks Start Interview
    timer_started_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # Total interview time elapsed, excluding all pause durations
    total_elapsed_secs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Cumulative pause duration across all disconnection events
    total_pause_secs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Timestamp of most recent disconnect event
    paused_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # Five minutes from paused_at — after which session is deactivated
    grace_period_expires_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )

    # Set to TRUE by the timer manager when the total interview timer expires.
    # Every LangGraph node checks this flag at entry and routes to
    # interview_complete_node if set, ensuring clean auto-submit regardless
    # of which node is currently active.
    auto_submit_triggered: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )

    # Ordered log of all disconnect and reconnect events with timestamps.
    # Each entry: {event, timestamp, pause_duration_secs}
    connectivity_events: Mapped[list] = mapped_column(
        JSONB, nullable=False, default=list, server_default="[]"
    )

    # ── Timestamps ────────────────────────────────────────────────────────────

    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    last_updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # ── Relationships ─────────────────────────────────────────────────────────

    transcript_turns: Mapped[list["TranscriptTurn"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="TranscriptTurn.turn_number",
    )
    answer_evaluations: Mapped[list["AnswerEvaluation"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="AnswerEvaluation.turn_number",
    )

    # Add this to InterviewSession relationships alongside transcript_turns and answer_evaluations
    evaluation: Mapped["InterviewEvaluation | None"] = relationship(
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return (
            f"<InterviewSession id={self.id} "
            f"status={self.status} section={self.current_section}>"
        )
