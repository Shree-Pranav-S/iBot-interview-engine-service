"""Build the complete immutable JSON context for one-shot evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from src.core.services.evaluation_errors import PermanentEvaluationError
from src.schemas.evaluation_llm import (
    EvaluationCandidateContext,
    EvaluationInput,
)

EVALUATION_SCHEMA_VERSION = "holistic-evaluation-v1"


@dataclass(frozen=True)
class TechnicalSkillSpec:
    name: str
    priority_score: float
    expected_signals: tuple[str, ...]
    questions_asked: int


@dataclass(frozen=True)
class EvaluationContextBundle:
    evaluation_input: EvaluationInput
    technical_skills: tuple[TechnicalSkillSpec, ...]


def _json_object(value: Any) -> dict[str, Any]:
    """
    Safely convert a value to a JSON-like dictionary object.

    Args:
        value: The value to convert.

    Returns:
        A dictionary representation of the value, or an empty dictionary if invalid.
    """
    return dict(value) if isinstance(value, dict) else {}


def _json_list(value: Any) -> list[Any]:
    """
    Safely convert a value to a list.

    Args:
        value: The value to convert.

    Returns:
        A list representation of the value, or an empty list if invalid.
    """
    return list(value) if isinstance(value, list) else []


def _skill_key(value: Any) -> str:
    """
    Normalize a skill name into a consistent lowercase string for matching.

    Args:
        value: The raw skill name.

    Returns:
        A normalized string containing only alphanumeric characters and allowed symbols (+, #, .).
    """
    return " ".join(re.findall(r"[a-z0-9+#.]+", str(value).casefold()))


def _score(value: Any, default: float = 1.0) -> float:
    """
    Safely parse a numerical score and clamp it between 0.0 and 10.0.

    Args:
        value: The value to parse as a float.
        default: The fallback value if parsing fails.

    Returns:
        A clamped float score between 0.0 and 10.0.
    """
    try:
        return min(10.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _priority_map(jd_analysis: dict[str, Any]) -> dict[str, float]:
    """
    Extract a mapping of skill names to their priority scores from the JD analysis.

    Args:
        jd_analysis: The JSON representation of the Job Description analysis.

    Returns:
        A dictionary mapping normalized skill keys to their priority scores.
    """
    priorities: dict[str, float] = {}
    for item in _json_list(jd_analysis.get("skills")):
        if not isinstance(item, dict):
            continue
        name = str(
            item.get("skill") or item.get("name") or item.get("title") or ""
        ).strip()
        if name:
            priorities[_skill_key(name)] = _score(
                item.get("priority_score"),
                default=1.0,
            )
    return priorities


def _priority_for_skill(
    skill: str,
    priorities: dict[str, float],
) -> float:
    """
    Determine the priority score for a specific skill using exact and partial matching.

    Args:
        skill: The skill name to lookup.
        priorities: A dictionary mapping normalized skill names to their scores.

    Returns:
        The priority score for the skill, defaulting to 1.0 if not found.
    """
    key = _skill_key(skill)
    if key in priorities:
        return priorities[key]
    for candidate, priority in priorities.items():
        if (
            candidate
            and key
            and min(len(candidate), len(key)) >= 3
            and (candidate in key or key in candidate)
        ):
            return priority
    return 1.0


def _question_counts(transcript: list[dict[str, Any]]) -> dict[str, int]:
    """
    Count the number of questions asked per technical skill by analyzing the transcript.

    Args:
        transcript: The list of transcript turns (messages).

    Returns:
        A dictionary mapping normalized skill keys to the number of distinct questions asked.
    """
    question_ids: dict[str, set[str]] = {}
    for turn in transcript:
        if str(turn.get("speaker") or "").casefold() != "bot":
            continue
        skill = str(
            turn.get("current_skill") or turn.get("skill") or turn.get("section") or ""
        ).strip()
        metadata = _json_object(turn.get("metadata"))
        question_type = str(
            turn.get("question_type") or metadata.get("question_type") or ""
        ).casefold()
        # Historical transcripts predate question_type but still carry the
        # active technical skill. Non-question response types are explicit,
        # so an empty type with a skill is a legacy scored question.
        if not skill or question_type not in {"", "new_question"}:
            continue
        question_id = str(
            turn.get("question_id")
            or metadata.get("question_id")
            or turn.get("turn_id")
            or len(question_ids.get(_skill_key(skill), set()))
        )
        question_ids.setdefault(_skill_key(skill), set()).add(question_id)
    return {key: len(values) for key, values in question_ids.items()}


def _technical_skill_specs(
    *,
    jd_analysis: dict[str, Any],
    interview_plan: dict[str, Any],
    transcript: list[dict[str, Any]],
) -> tuple[TechnicalSkillSpec, ...]:
    """
    Construct a definitive list of technical skills to be evaluated.

    Args:
        jd_analysis: The structured JD analysis dictionary.
        interview_plan: The structured interview plan dictionary.
        transcript: The list of transcript turns.

    Returns:
        A tuple of TechnicalSkillSpec objects representing the skills to evaluate.

    Raises:
        PermanentEvaluationError: If no technical skills could be inferred.
    """
    priorities = _priority_map(jd_analysis)
    counts = _question_counts(transcript)
    specs: list[TechnicalSkillSpec] = []
    seen: set[str] = set()

    for section in _json_list(interview_plan.get("sections")):
        if not isinstance(section, dict):
            continue
        name = str(section.get("skill") or section.get("section_name") or "").strip()
        section_name = str(section.get("section_name") or "").casefold()
        if not name or section_name in {"self_intro", "behavioural_cultural"}:
            continue
        key = _skill_key(name)
        if not key or key in seen:
            continue
        expected_signals = tuple(
            str(item).strip()
            for item in _json_list(section.get("expected_signals"))
            if str(item).strip()
        )
        specs.append(
            TechnicalSkillSpec(
                name=name,
                priority_score=_priority_for_skill(name, priorities),
                expected_signals=expected_signals,
                questions_asked=counts.get(key, 0),
            )
        )
        seen.add(key)

    if not specs:
        for item in _json_list(jd_analysis.get("skills")):
            if not isinstance(item, dict):
                continue
            name = str(item.get("skill") or item.get("name") or "").strip()
            key = _skill_key(name)
            if not name or key in seen:
                continue
            specs.append(
                TechnicalSkillSpec(
                    name=name,
                    priority_score=_priority_for_skill(name, priorities),
                    expected_signals=(),
                    questions_asked=counts.get(key, 0),
                )
            )
            seen.add(key)
    if not specs:
        raise PermanentEvaluationError(
            "Evaluation requires at least one planned technical skill"
        )
    return tuple(specs)


def _transcript_hash(
    *,
    transcript: list[dict[str, Any]],
    violations: list[dict[str, Any]],
    jd_analysis: dict[str, Any],
    interview_plan: dict[str, Any],
) -> str:
    """
    Compute a SHA-256 fingerprint of the evaluation context.

    Args:
        transcript: The list of transcript turns.
        violations: Any proctoring violations recorded during the interview.
        jd_analysis: The parsed JD analysis.
        interview_plan: The structured interview plan.

    Returns:
        A hexadecimal string representing the deterministic hash.
    """
    canonical = json.dumps(
        {
            "transcript": transcript,
            "violations": violations,
            "jd_analysis": jd_analysis,
            "interview_plan": interview_plan,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_evaluation_context(
    source: dict[str, Any],
) -> EvaluationContextBundle:
    """
    Validate repository data, extract skills, calculate properties, and build
    the complete immutable JSON context bundle for one-shot evaluation.

    Args:
        source: The raw source data dictionary retrieved from the database.

    Returns:
        An EvaluationContextBundle containing validated evaluation inputs and skills.

    Raises:
        PermanentEvaluationError: If the source data lacks critical information like transcript or interview plan.
    """

    transcript = [
        dict(item)
        for item in _json_list(source.get("transcript"))
        if isinstance(item, dict)
    ]
    if not transcript:
        raise PermanentEvaluationError(
            "Cannot evaluate an interview with an empty transcript"
        )
    violations = [
        dict(item)
        for item in _json_list(source.get("violations"))
        if isinstance(item, dict)
    ]
    jd_analysis = _json_object(source.get("jd_analysis"))
    interview_plan = _json_object(source.get("interview_plan"))
    if not interview_plan:
        raise PermanentEvaluationError("Interview plan is missing")

    technical_skills = _technical_skill_specs(
        jd_analysis=jd_analysis,
        interview_plan=interview_plan,
        transcript=transcript,
    )
    inferred_difficulty = str(
        interview_plan.get("inferred_difficulty")
        or jd_analysis.get("inferred_difficulty")
        or "mid-level"
    ).strip()
    candidate = EvaluationCandidateContext(
        candidate_assessment_id=str(source["candidate_assessment_id"]),
        session_id=str(source["session_id"]),
        assessment_id=str(source["assessment_id"]),
        candidate_name=str(source.get("candidate_name") or "Candidate"),
        candidate_email=(
            str(source["candidate_email"]) if source.get("candidate_email") else None
        ),
        assessment_title=str(source.get("assessment_title") or "Interview assessment"),
        role_name=str(source.get("role_name") or "the role"),
        company_name=str(source.get("company_name") or "the company"),
        inferred_difficulty=inferred_difficulty,
        interview_duration_mins=max(
            1,
            int(source.get("interview_duration_mins") or 1),
        ),
        total_elapsed_secs=max(
            0,
            int(source.get("total_elapsed_secs") or 0),
        ),
    )
    evaluation_input = EvaluationInput(
        evaluation_schema_version=EVALUATION_SCHEMA_VERSION,
        transcript_hash=_transcript_hash(
            transcript=transcript,
            violations=violations,
            jd_analysis=jd_analysis,
            interview_plan=interview_plan,
        ),
        candidate=candidate,
        jd_analysis=jd_analysis,
        interview_plan=interview_plan,
        transcript=transcript,
        violations=violations,
    )
    return EvaluationContextBundle(
        evaluation_input=evaluation_input,
        technical_skills=technical_skills,
    )
