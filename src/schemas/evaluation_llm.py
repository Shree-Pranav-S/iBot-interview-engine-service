"""Strict contracts for one-shot holistic interview evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

HiringRecommendation = Literal["hire", "consider", "no hire"]
ViolationSeverity = Literal["low", "medium", "high", "critical"]


class EvaluationModel(BaseModel):
    """Reject unknown model output while allowing normal JSON number parsing."""

    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before", check_fields=False)
    @classmethod
    def clamp_model_scores(
        cls,
        value: Any,
        info: ValidationInfo,
    ) -> Any:
        """Apply the backend score-range safeguard before field validation."""

        name = info.field_name
        upper_bound = (
            1.0
            if name == "confidence"
            else 10.0
            if name == "score" or name.endswith("_score")
            else None
        )
        if upper_bound is None or isinstance(value, bool):
            return value
        try:
            return min(upper_bound, max(0.0, float(value)))
        except (TypeError, ValueError):
            return value


class EvaluationCandidateContext(EvaluationModel):
    candidate_assessment_id: str
    session_id: str
    assessment_id: str
    candidate_name: str
    candidate_email: str | None = None
    assessment_title: str
    role_name: str
    company_name: str
    inferred_difficulty: str
    interview_duration_mins: int
    total_elapsed_secs: int


class EvaluationInput(EvaluationModel):
    """Complete immutable context sent to the evaluator in one request."""

    evaluation_schema_version: str
    transcript_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate: EvaluationCandidateContext
    jd_analysis: dict[str, Any]
    interview_plan: dict[str, Any]
    transcript: list[dict[str, Any]] = Field(min_length=1)
    violations: list[dict[str, Any]]


class SkillScoreOutput(EvaluationModel):
    score: float = Field(ge=0.0, le=10.0)
    priority_score: float = Field(ge=0.0, le=10.0)
    questions_evaluated: int = Field(ge=0)
    confidence: float = Field(ge=0.0, le=1.0)


class SectionCommunicationOutput(EvaluationModel):
    score: float = Field(ge=0.0, le=10.0)
    summary: str = Field(min_length=1, max_length=1800)
    evidence: list[str] = Field(default_factory=list, max_length=16)


class SectionCommunicationScores(EvaluationModel):
    self_intro: SectionCommunicationOutput
    technical: SectionCommunicationOutput
    behavioural_cultural: SectionCommunicationOutput


class SeverityCounts(EvaluationModel):
    low: int = Field(ge=0)
    medium: int = Field(ge=0)
    high: int = Field(ge=0)
    critical: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.low + self.medium + self.high + self.critical


class ViolationSummaryOutput(EvaluationModel):
    has_violation: bool
    validated_violation_count: int = Field(ge=0)
    severity_counts: SeverityCounts
    summary: str = Field(min_length=1, max_length=2200)

    @model_validator(mode="after")
    def validate_counts(self) -> ViolationSummaryOutput:
        if self.severity_counts.total != self.validated_violation_count:
            raise ValueError("severity_counts must sum to validated_violation_count")
        if self.has_violation != (self.validated_violation_count > 0):
            raise ValueError("has_violation must match validated_violation_count")
        return self


class HolisticEvaluationLLMOutput(EvaluationModel):
    """The only JSON object accepted from the holistic evaluator."""

    intro_section_score: float = Field(ge=0.0, le=10.0)
    intro_section_summary: str = Field(min_length=1, max_length=2400)
    intro_section_evidence: list[str] = Field(
        default_factory=list,
        max_length=16,
    )

    skill_scores: dict[str, SkillScoreOutput] = Field(min_length=1)
    skill_summary: dict[str, str] = Field(min_length=1)
    skill_evidence: dict[str, list[str]] = Field(min_length=1)
    overall_technical_skill_score: float = Field(ge=0.0, le=10.0)

    behavioural_cultural_score: float = Field(ge=0.0, le=10.0)
    behavioural_cultural_summary: str = Field(
        min_length=1,
        max_length=2600,
    )
    behavioural_cultural_evidence: list[str] = Field(
        min_length=1,
        max_length=20,
    )

    communication_score: float = Field(ge=0.0, le=10.0)
    communication_summary: str = Field(min_length=1, max_length=2400)
    communication_evidence: list[str] = Field(min_length=1, max_length=20)
    section_communication_scores: SectionCommunicationScores

    violation_summary: ViolationSummaryOutput
    violation_evidence: list[str] = Field(default_factory=list, max_length=24)

    overall_score: float = Field(ge=0.0, le=10.0)
    hiring_recommendation: HiringRecommendation
    overall_summary: str = Field(min_length=1, max_length=3200)
    recommendation_reasoning: str = Field(min_length=1, max_length=3200)

    strengths: list[str] = Field(default_factory=list, max_length=24)
    concerns: list[str] = Field(default_factory=list, max_length=24)

    @model_validator(mode="after")
    def validate_evidence_and_skill_maps(
        self,
    ) -> HolisticEvaluationLLMOutput:
        score_keys = set(self.skill_scores)
        if set(self.skill_summary) != score_keys:
            raise ValueError("skill_summary keys must exactly match skill_scores keys")
        if set(self.skill_evidence) != score_keys:
            raise ValueError("skill_evidence keys must exactly match skill_scores keys")

        for skill, details in self.skill_scores.items():
            summary = str(self.skill_summary.get(skill) or "").strip()
            evidence = self.skill_evidence.get(skill) or []
            if not summary:
                raise ValueError(f"{skill} requires a non-empty summary")
            if details.questions_evaluated > 0 and not evidence:
                raise ValueError(
                    f"{skill} was evaluated but has no transcript evidence"
                )
            if details.score >= 8.0 and not evidence:
                raise ValueError(
                    f"{skill} has a high score without transcript evidence"
                )

        if self.intro_section_score >= 7.0 and not self.intro_section_evidence:
            raise ValueError("high intro_section_score requires transcript evidence")
        if (
            self.violation_summary.validated_violation_count > 0
            and not self.violation_evidence
        ):
            raise ValueError("validated violations require violation_evidence")
        return self


class FinalEvaluationRecord(EvaluationModel):
    """Deterministically scored record ready for database persistence."""

    candidate_assessment_id: str
    session_id: str
    assessment_id: str

    intro_section_score: float
    intro_section_summary: str
    intro_section_evidence: list[str]

    skill_scores: dict[str, dict[str, Any]]
    overall_technical_skill_score: float
    skill_summary: dict[str, str]
    skill_evidence: dict[str, list[str]]

    behavioural_cultural_score: float
    behavioural_cultural_summary: str
    behavioural_cultural_evidence: list[str]

    communication_score: float
    communication_summary: str
    communication_evidence: list[str]
    section_communication_scores: dict[str, Any]

    violation_summary: dict[str, Any]
    violation_evidence: list[str]

    raw_overall_score: float
    violation_penalty: float
    overall_score: float
    hiring_recommendation: HiringRecommendation
    model_recommendation: HiringRecommendation
    recommendation_override_reason: str | None
    overall_summary: str
    recommendation_reasoning: str
    strengths: list[str]
    concerns: list[str]

    prompt_version: str
    model_name: str
    model_provider: str
    evaluation_schema_version: str
    transcript_hash: str
    raw_model_output: dict[str, Any]
