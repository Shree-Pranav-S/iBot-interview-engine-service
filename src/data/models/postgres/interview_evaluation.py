"""Persisted holistic interview evaluation and audit metadata."""

import uuid
from datetime import datetime

from sqlalchemy import Float, Integer, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from src.data.models.postgres.base import Base


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
        nullable=False,
        unique=True,
    )

    intro_section_score: Mapped[float] = mapped_column(Float, nullable=False)
    intro_section_summary: Mapped[str] = mapped_column(Text, nullable=False)
    intro_section_evidence: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
    )
    skill_scores: Mapped[dict] = mapped_column(JSONB, nullable=False)
    overall_technical_skill_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    skill_summary: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    skill_evidence: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    behavioural_cultural_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    behavioural_cultural_summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    behavioural_cultural_evidence: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
    )
    communication_score: Mapped[float] = mapped_column(Float, nullable=False)
    communication_summary: Mapped[str] = mapped_column(Text, nullable=False)
    communication_evidence: Mapped[list[str]] = mapped_column(
        ARRAY(Text),
        nullable=False,
    )
    section_communication_scores: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        server_default="{}",
    )
    violation_summary: Mapped[dict | None] = mapped_column(
        JSONB,
        nullable=True,
    )
    violation_evidence: Mapped[list[str] | None] = mapped_column(
        ARRAY(Text),
        nullable=True,
    )
    raw_overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    violation_penalty: Mapped[float] = mapped_column(Float, nullable=False)
    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    hiring_recommendation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )
    model_recommendation: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation_override_reason: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    overall_summary: Mapped[str] = mapped_column(Text, nullable=False)
    recommendation_reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    strengths: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    concerns: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    prompt_version: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(Text, nullable=False)
    model_provider: Mapped[str] = mapped_column(Text, nullable=False)
    evaluation_schema_version: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    transcript_hash: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )
    raw_model_output: Mapped[dict] = mapped_column(JSONB, nullable=False)
    rank_in_assessment: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    percentile_in_assessment: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    total_candidates_evaluated: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )
    generated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return (
            "<InterviewEvaluation "
            f"candidate_assessment_id={self.candidate_assessment_id} "
            f"score={self.overall_score} "
            f"recommendation={self.hiring_recommendation}>"
        )
