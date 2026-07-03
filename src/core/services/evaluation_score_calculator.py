"""Deterministic scoring, penalties, gates, and recommendation overrides."""

from __future__ import annotations

from typing import Any

from src.config.settings import settings
from src.core.services.evaluation_context_builder import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationContextBundle,
)
from src.core.services.evaluation_llm_client import NvidiaEvaluationResult
from src.schemas.evaluation_llm import FinalEvaluationRecord

TECHNICAL_WEIGHT = 0.75
BEHAVIOURAL_CULTURAL_WEIGHT = 0.15
INTRO_WEIGHT = 0.05
COMMUNICATION_WEIGHT = 0.05
PRIORITY_EXPONENT = 1.35
HIGH_PRIORITY_THRESHOLD = 7.0

QUESTION_DIFFICULTY_WEIGHTS: dict[str, float] = {
    "easy": 1.0,
    "medium": 1.1,
    "hard": 1.15,
}
RELEVANCE_SCORE_CAPS: dict[str, float] = {
    "direct_match": 10.0,
    "close_equivalent": 8.0,
    "transferable_similar": 6.5,
    "adjacent_but_not_equivalent": 4.5,
    "unrelated": 2.0,
    "not_applicable": 10.0,
}

PENALTY_BY_SEVERITY: dict[str, float] = {
    "low": 0.15,
    "medium": 0.40,
    "high": 1.00,
    "critical": 2.00,
}


def _clamp_score(value: float) -> float:
    """
    Ensure a score is within the allowed boundaries (0.0 to 10.0).

    Args:
        value: The raw float score.

    Returns:
        The score clamped to [0.0, 10.0].
    """
    return min(10.0, max(0.0, float(value)))


def _round_score(value: float) -> float:
    """
    Clamp a score to boundaries and round it to two decimal places.

    Args:
        value: The raw float score.

    Returns:
        The rounded and clamped score.
    """
    return round(_clamp_score(value), 2)


def _technical_score(
    skill_scores: dict[str, dict[str, Any]],
) -> float:
    """
    Calculate the overall technical score as a weighted average of individual skill scores.
    The weight is determined by applying an exponent to the skill's priority score.

    Args:
        skill_scores: The dictionary mapping skill names to their score details.

    Returns:
        The calculated weighted technical score (0.0 to 10.0).
    """
    weighted_total = 0.0
    weight_total = 0.0
    for details in skill_scores.values():
        if int(details.get("questions_evaluated") or 0) <= 0:
            continue
        score = _clamp_score(float(details.get("score") or 0.0))
        priority = max(0.0, float(details.get("priority_score") or 0.0))
        weight = priority**PRIORITY_EXPONENT
        if weight <= 0:
            continue
        weighted_total += score * weight
        weight_total += weight
    return weighted_total / weight_total if weight_total else 0.0


def _authoritative_question_evaluations(
    model_result: NvidiaEvaluationResult,
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for item in model_result.output.question_evaluations:
        value = item.model_dump(mode="json")
        relevance_cap = RELEVANCE_SCORE_CAPS[item.relevance_class]
        value["score"] = _round_score(min(item.score, relevance_cap))
        value["confidence"] = round(
            min(1.0, max(0.0, item.confidence)),
            3,
        )
        evaluations.append(value)
    return evaluations


def _skill_question_aggregate(
    *,
    skill: str,
    question_evaluations: list[dict[str, Any]],
) -> tuple[float, float] | None:
    matching = [
        item
        for item in question_evaluations
        if str(item.get("skill") or "").casefold() == skill.casefold()
    ]
    if not matching:
        return None

    weighted_score = 0.0
    weighted_confidence = 0.0
    total_weight = 0.0
    for item in matching:
        difficulty = str(item.get("difficulty") or "unknown").casefold()
        weight = QUESTION_DIFFICULTY_WEIGHTS.get(difficulty, 1.0)
        weighted_score += _clamp_score(float(item.get("score") or 0.0)) * weight
        weighted_confidence += (
            min(
                1.0,
                max(0.0, float(item.get("confidence") or 0.0)),
            )
            * weight
        )
        total_weight += weight
    if total_weight <= 0:
        return 0.0, 0.0
    return weighted_score / total_weight, weighted_confidence / total_weight


def _has_behavioural_questions(bundle: EvaluationContextBundle) -> bool:
    return any(
        "behav" in item.section.casefold() or "cultur" in item.section.casefold()
        for item in bundle.evaluation_input.qa_pairs
    )


def _violation_penalty(severity_counts: dict[str, int]) -> float:
    """
    Calculate the total score penalty to apply based on the number and severity of violations.

    Args:
        severity_counts: A dictionary mapping severity levels to the count of occurrences.

    Returns:
        The total float penalty to subtract from the final score.
    """
    return sum(
        max(0, int(severity_counts.get(severity) or 0)) * penalty
        for severity, penalty in PENALTY_BY_SEVERITY.items()
    )


def _recommendation(
    *,
    final_score: float,
    technical_score: float,
    validated_violation_count: int,
    critical_violation_count: int,
    skill_scores: dict[str, dict[str, Any]],
) -> tuple[str, list[str]]:
    """
    Determine the final hiring recommendation based on strict deterministic thresholds.
    Applies hard gates (e.g., critical violations, low technical score) to override the LLM.

    Args:
        final_score: The overall calculated score (0-10).
        technical_score: The calculated technical skill score (0-10).
        validated_violation_count: The total number of valid violations.
        critical_violation_count: The number of critical violations.
        skill_scores: The individual skill scores to check for high-priority failures.

    Returns:
        A tuple containing the recommendation string ('hire', 'consider', 'no hire')
        and a list of any hard-gate reasons triggered.
    """
    gates: list[str] = []
    if validated_violation_count > 7:
        gates.append("More than seven violations were validated")
    if critical_violation_count > 0:
        gates.append("At least one critical violation was validated")
    if technical_score < 5.0:
        gates.append("Overall technical skill score is below 5.0")

    for skill, details in skill_scores.items():
        if int(details.get("questions_evaluated") or 0) <= 0:
            continue
        priority = float(details.get("priority_score") or 0.0)
        score = float(details.get("score") or 0.0)
        if priority >= HIGH_PRIORITY_THRESHOLD and score < 4.0:
            gates.append(f"High-priority skill {skill} scored below 4.0")

    if gates:
        return "no hire", gates
    if final_score >= 7.5 and technical_score >= 7.0:
        return "hire", []
    if final_score >= 5.5:
        return "consider", []
    return "no hire", ["Final score is below 5.5"]


def _strengths_and_concerns(
    skill_scores: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """
    Identify notable strengths and concerns based on the individual skill scores.

    Args:
        skill_scores: The individual skill scores.

    Returns:
        A tuple of two lists: strengths (skills scoring >= 7.5) and concerns (skills scoring < 5.5).
    """
    strengths: list[str] = []
    concerns: list[str] = []
    for skill, details in skill_scores.items():
        if int(details.get("questions_evaluated") or 0) <= 0:
            continue
        score = float(details.get("score") or 0.0)
        priority = float(details.get("priority_score") or 0.0)
        if score >= 7.0:
            strengths.append(skill)
        if score < 5.0 or (priority >= HIGH_PRIORITY_THRESHOLD and score < 5.5):
            concerns.append(skill)
    return strengths, concerns


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
        hiring_recommendation=final_recommendation,  # type: ignore[arg-type]
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
