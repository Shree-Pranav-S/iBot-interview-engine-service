"""Internal schemas for holistic evaluation processing."""

from __future__ import annotations

from dataclasses import dataclass

from src.schemas.evaluation_llm import EvaluationInput

EVALUATION_SCHEMA_VERSION = "holistic-evaluation-v3"


@dataclass(frozen=True)
class TechnicalSkillSpec:
    """Authoritative metadata for one planned technical skill."""

    name: str
    priority_score: float
    expected_signals: tuple[str, ...]
    questions_asked: int = 0


@dataclass(frozen=True)
class EvaluationContextBundle:
    """Validated evaluator input paired with authoritative skill metadata."""

    evaluation_input: EvaluationInput
    technical_skills: tuple[TechnicalSkillSpec, ...]
