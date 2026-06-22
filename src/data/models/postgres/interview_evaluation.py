import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.data.models.postgres.base import Base

if TYPE_CHECKING:
    from src.data.models.postgres.interview_session import InterviewSession


class InterviewEvaluation(Base):
    __tablename__ = "interview_evaluations"

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
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("interview_sessions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    skill_scores: Mapped[dict] = mapped_column(JSONB, nullable=False)
    technical_dimension_score: Mapped[float] = mapped_column(Float, nullable=False)
    score_evidence: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    score_summary: Mapped[str] = mapped_column(Text, nullable=False)

    behavioural_score: Mapped[float] = mapped_column(Float, nullable=False)
    behavioural_evidence: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    behavioural_summary: Mapped[str] = mapped_column(Text, nullable=False)

    cultural_fit_score: Mapped[float] = mapped_column(Float, nullable=False)
    cultural_fit_evidence: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False
    )
    cultural_fit_summary: Mapped[str] = mapped_column(Text, nullable=False)

    tone_classification_score: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    tone_distribution: Mapped[list | None] = mapped_column(JSONB, nullable=True)

    section_summaries: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )

    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    hiring_recommendation: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    overall_narrative: Mapped[str] = mapped_column(Text, nullable=False)
    strengths: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    concerns: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    violation_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    best_answer: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    weakest_answer: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    recommendation_reasoning: Mapped[str] = mapped_column(Text, nullable=False)

    rank_in_assessment: Mapped[int | None] = mapped_column(Integer, nullable=True)
    percentile_in_assessment: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_candidates_evaluated: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    generated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    session: Mapped["InterviewSession"] = relationship(
        back_populates="evaluation",
        uselist=False,
    )

    def __repr__(self) -> str:
        return (
            f"<InterviewEvaluation candidate_assessment_id={self.candidate_assessment_id} "
            f"score={self.overall_score} recommendation={self.hiring_recommendation}>"
        )
