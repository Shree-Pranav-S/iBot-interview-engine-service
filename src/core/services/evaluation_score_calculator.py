"""Deterministic scoring, penalties, gates, and recommendation overrides."""

from __future__ import annotations

from typing import Any

from src.config.settings import settings
from src.core.services.evaluation_llm_client import NvidiaEvaluationResult
from src.schemas.evaluation_llm import FinalEvaluationRecord
from src.schemas.internal_evaluation import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationContextBundle,
)
from src.utils.evaluation_scoring import (
    BEHAVIOURAL_CULTURAL_WEIGHT,
    COMMUNICATION_WEIGHT,
    INTRO_WEIGHT,
    TECHNICAL_WEIGHT,
    _authoritative_question_evaluations,
    _clamp_score,
    _has_behavioural_questions,
    _recommendation,
    _round_score,
    _skill_question_aggregate,
    _strengths_and_concerns,
    _technical_score,
    _violation_penalty,
)


def calculate_final_evaluation(
    *,
    bundle: EvaluationContextBundle,
    model_result: NvidiaEvaluationResult,
) -> FinalEvaluationRecord:
    """
    Replace model aggregates with authoritative deterministic calculations.

    This acts as the final decision layer. It ignores the LLM's own calculation
    of the overall score, recalculates it using proper priority weighting, applies
    violation penalties, and enforces deterministic hiring gates.

    Args:
        bundle: The original evaluation context used to prompt the LLM.
        model_result: The parsed output from the LLM evaluation.

    Returns:
        A FinalEvaluationRecord ready to be persisted and presented to the recruiter.
    """

    output = model_result.output
    question_evaluations = _authoritative_question_evaluations(model_result)
    authoritative_specs = {item.name: item for item in bundle.technical_skills}
    skill_scores: dict[str, dict[str, Any]] = {}
    for skill, model_details in output.skill_scores.items():
        spec = authoritative_specs[skill]
        question_aggregate = _skill_question_aggregate(
            skill=skill,
            question_evaluations=question_evaluations,
        )
        if question_aggregate is None:
            aggregate_score = model_details.score
            aggregate_confidence = model_details.confidence
        else:
            aggregate_score, aggregate_confidence = question_aggregate
        skill_scores[skill] = {
            "score": _round_score(aggregate_score),
            "priority_score": round(spec.priority_score, 2),
            "questions_evaluated": spec.questions_asked,
            "confidence": round(
                min(1.0, max(0.0, aggregate_confidence)),
                3,
            ),
        }

    technical_score = _technical_score(skill_scores)
    intro_score = _clamp_score(output.intro_section_score)
    behavioural_score = _clamp_score(output.behavioural_cultural_score)
    severity_counts = output.violation_summary.severity_counts.model_dump()
    # If the behavioural section was never reached, lack of evidence is not
    # negative candidate evidence. Preserve a neutral floor unless validated
    # serious conduct elsewhere supports a lower score.
    if (
        not _has_behavioural_questions(bundle)
        and not severity_counts["high"]
        and not severity_counts["critical"]
    ):
        behavioural_score = max(5.0, behavioural_score)
    communication_score = _clamp_score(output.communication_score)
    raw_overall_score = (
        technical_score * TECHNICAL_WEIGHT
        + behavioural_score * BEHAVIOURAL_CULTURAL_WEIGHT
        + intro_score * INTRO_WEIGHT
        + communication_score * COMMUNICATION_WEIGHT
    )

    violation_penalty = _violation_penalty(severity_counts)
    final_score = max(0.0, raw_overall_score - violation_penalty)
    final_recommendation, gates = _recommendation(
        final_score=final_score,
        technical_score=technical_score,
        validated_violation_count=(output.violation_summary.validated_violation_count),
        critical_violation_count=severity_counts["critical"],
        skill_scores=skill_scores,
    )
    model_recommendation = output.hiring_recommendation
    override_reason = None
    if final_recommendation != model_recommendation:
        override_reason = "; ".join(gates) or (
            "Deterministic score thresholds changed the model recommendation"
        )

    strengths, concerns = _strengths_and_concerns(skill_scores)
    violation_summary = output.violation_summary.model_dump(mode="json")
    violation_summary.update(
        {
            "penalty_applied": round(violation_penalty, 2),
            "hard_gate_reasons": gates,
        }
    )
    recommendation_reasoning = output.recommendation_reasoning.strip()
    if override_reason:
        recommendation_reasoning = (
            f"{recommendation_reasoning} Deterministic override: {override_reason}."
        )

    return FinalEvaluationRecord(
        candidate_assessment_id=(
            bundle.evaluation_input.candidate.candidate_assessment_id
        ),
        session_id=bundle.evaluation_input.candidate.session_id,
        assessment_id=bundle.evaluation_input.candidate.assessment_id,
        intro_section_score=_round_score(intro_score),
        intro_section_summary=output.intro_section_summary,
        intro_section_evidence=output.intro_section_evidence,
        skill_scores=skill_scores,
        overall_technical_skill_score=_round_score(technical_score),
        skill_summary=output.skill_summary,
        skill_evidence=output.skill_evidence,
        question_evaluations=question_evaluations,
        behavioural_cultural_score=_round_score(behavioural_score),
        behavioural_cultural_summary=(output.behavioural_cultural_summary),
        behavioural_cultural_evidence=(output.behavioural_cultural_evidence),
        communication_score=_round_score(communication_score),
        communication_summary=output.communication_summary,
        communication_evidence=output.communication_evidence,
        section_communication_scores=(
            output.section_communication_scores.model_dump(mode="json")
        ),
        violation_summary=violation_summary,
        violation_evidence=output.violation_evidence,
        raw_overall_score=round(raw_overall_score, 2),
        violation_penalty=round(violation_penalty, 2),
        overall_score=_round_score(final_score),
        hiring_recommendation=final_recommendation,
        model_recommendation=model_recommendation,
        recommendation_override_reason=override_reason,
        overall_summary=output.overall_summary,
        recommendation_reasoning=recommendation_reasoning,
        strengths=strengths,
        concerns=concerns,
        prompt_version="nvidia-nemotron-single-stage-v1",
        model_name=settings.NVIDIA_NIM_MODEL,
        model_provider="nvidia_nim",
        evaluation_schema_version=EVALUATION_SCHEMA_VERSION,
        transcript_hash=bundle.evaluation_input.transcript_hash,
        raw_model_output=model_result.raw_output,
    )
