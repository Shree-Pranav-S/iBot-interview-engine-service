"""Evaluation schemas for holistic interview reporting."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictEvaluationModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TranscriptEvidence(StrictEvaluationModel):
    turn_number: int | None = Field(default=None, ge=0)
    section: str | None = None
    skill: str | None = None
    quote: str = Field(min_length=1, max_length=900)
    interpretation: str = Field(min_length=1, max_length=900)


class SkillScoreReport(StrictEvaluationModel):
    priority_score: float | None = Field(default=None, ge=0.0, le=10.0)
    depth_required: str | None = None
    raw_score: float | None = Field(default=None, ge=0.0, le=10.0)
    weighted_score: float = Field(ge=0.0, le=100.0)
    weight_share: float | None = Field(default=None, ge=0.0, le=100.0)
    weighted_contribution: float | None = Field(default=None, ge=0.0, le=100.0)
    difficulty_reached: str | None = None
    questions_asked: int = Field(ge=0)
    assessed: bool
    similar_skill_credit: bool = False
    similar_skills_considered: list[str] = Field(default_factory=list, max_length=8)
    transcript_evidence: list[TranscriptEvidence] = Field(
        default_factory=list,
        max_length=12,
    )
    signals_demonstrated: list[str] = Field(default_factory=list, max_length=16)
    signals_missing: list[str] = Field(default_factory=list, max_length=16)
    summary: str = Field(min_length=1, max_length=1800)


class SectionSummaryReport(StrictEvaluationModel):
    summary: str = Field(min_length=1, max_length=1400)
    avg_score: float | None = Field(default=None, ge=0.0, le=10.0)
    difficulty_reached: str | None = None
    questions_asked: int = Field(ge=0)
    evidence: list[TranscriptEvidence] = Field(default_factory=list, max_length=8)


class ViolationSummaryReport(StrictEvaluationModel):
    total_irrelevant: int = Field(ge=0)
    total_silences: int = Field(ge=0)
    terminated_early: bool
    entries: list[dict[str, Any]] = Field(default_factory=list)


class HighlightAnswerReport(StrictEvaluationModel):
    question: str = Field(min_length=1, max_length=1200)
    turn_number: int = Field(ge=0)
    section: str | None = None
    difficulty_at_time: str | None = None
    reason: str = Field(min_length=1, max_length=1200)


class HolisticEvaluationReport(StrictEvaluationModel):
    skill_scores: dict[str, SkillScoreReport] = Field(min_length=1)
    technical_dimension_score: float = Field(ge=0.0, le=100.0)
    score_evidence: list[str] = Field(min_length=1, max_length=20)
    score_summary: str = Field(min_length=1, max_length=2200)
    behavioural_score: float = Field(ge=0.0, le=100.0)
    behavioural_evidence: list[str] = Field(min_length=1, max_length=16)
    behavioural_summary: str = Field(min_length=1, max_length=1600)
    cultural_fit_score: float = Field(ge=0.0, le=100.0)
    cultural_fit_evidence: list[str] = Field(min_length=1, max_length=16)
    cultural_fit_summary: str = Field(min_length=1, max_length=1600)
    tone_classification_score: float | None = Field(default=None, ge=0.0, le=100.0)
    tone_distribution: list[dict[str, Any]] | None = None
    section_summaries: dict[str, SectionSummaryReport] = Field(min_length=1)
    overall_score: float = Field(ge=0.0, le=100.0)
    hiring_recommendation: Literal[
        "STRONG_HIRE",
        "HIRE",
        "CONSIDER",
        "WEAK",
        "NO_HIRE",
    ]
    overall_narrative: str = Field(min_length=1, max_length=2400)
    strengths: list[str] = Field(min_length=1, max_length=12)
    concerns: list[str] = Field(min_length=1, max_length=12)
    violation_summary: ViolationSummaryReport | None = None
    best_answer: HighlightAnswerReport | None = None
    weakest_answer: HighlightAnswerReport | None = None
    recommendation_reasoning: str = Field(min_length=1, max_length=2400)

    @model_validator(mode="after")
    def ensure_evidence_backed_report(self) -> "HolisticEvaluationReport":
        for skill, data in self.skill_scores.items():
            if data.assessed and not data.transcript_evidence:
                raise ValueError(f"{skill} is assessed but has no transcript_evidence")
        if self.violation_summary and not self.violation_summary.entries:
            raise ValueError("violation_summary requires entries when present")
        return self
