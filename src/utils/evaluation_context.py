"""Private helpers for evaluation context and Q&A pairing."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from src.core.exceptions.evaluation import PermanentEvaluationError
from src.schemas.evaluation_llm import QuestionAnswerPair
from src.schemas.internal_evaluation import TechnicalSkillSpec


def _normalize_skill_key(value: Any) -> str:
    """Normalize a skill name into a consistent lowercase matching key."""
    return " ".join(re.findall(r"[a-z0-9+#.]+", str(value).casefold()))


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
            priorities[_normalize_skill_key(name)] = _score(
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
    matched = _matched_priority_for_skill(skill, priorities)
    return matched if matched is not None else 1.0


def _matched_priority_for_skill(
    skill: str,
    priorities: dict[str, float],
) -> float | None:
    """Return an exact/partial JD priority match without applying a fallback."""
    key = _normalize_skill_key(skill)
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
    return None


def _time_aware_manual_priority(
    allocated_mins: float,
    average_topic_mins: float,
) -> float:
    """Give a recruiter-added topic a neutral priority scaled by its time share."""
    if allocated_mins <= 0 or average_topic_mins <= 0:
        return 1.0
    return min(10.0, max(1.0, 5.0 * allocated_mins / average_topic_mins))


def _allocated_minutes(value: Any) -> float:
    """Parse a non-negative plan duration without applying score clamping."""
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _string_list(value: Any) -> list[str]:
    return [str(item).strip() for item in _json_list(value) if str(item).strip()]


def _turn_value(turn: dict[str, Any], key: str) -> Any:
    value = turn.get(key)
    if value not in (None, ""):
        return value
    return _json_object(turn.get("metadata")).get(key)


def _question_id(turn: dict[str, Any]) -> str:
    return str(_turn_value(turn, "question_id") or "").strip()


def _response_type(turn: dict[str, Any]) -> str:
    metadata = _json_object(turn.get("metadata"))
    classification = _json_object(metadata.get("classification"))
    return str(
        turn.get("response_type")
        or metadata.get("response_type")
        or classification.get("response_type")
        or ""
    ).strip()


def _live_evaluation(turn: dict[str, Any]) -> dict[str, Any]:
    metadata = _json_object(turn.get("metadata"))
    value = turn.get("live_evaluation") or metadata.get("live_evaluation")
    return _json_object(value)


def _is_scored_question(turn: dict[str, Any]) -> bool:
    if str(turn.get("speaker") or "").casefold() != "bot":
        return False
    metadata = _json_object(turn.get("metadata"))
    question_type = str(
        turn.get("question_type")
        or metadata.get("question_type")
        or metadata.get("message_type")
        or ""
    ).casefold()
    if question_type in {"new_question", "question", "opening"}:
        return True
    if question_type:
        return False
    # Legacy transcripts only marked a bot turn with the active skill/section.
    return bool(
        turn.get("current_skill")
        or turn.get("skill")
        or turn.get("current_section")
        or turn.get("section")
    )


def _meaningful_answer(text: str, response_type: str) -> bool:
    normalized_type = response_type.casefold()
    if normalized_type in {"silence", "irrelevant", "refusal", "clarification"}:
        return False
    return bool(text.strip())


def _technical_skill_specs(
    *,
    jd_analysis: dict[str, Any],
    interview_plan: dict[str, Any],
) -> tuple[TechnicalSkillSpec, ...]:
    """
    Construct a definitive list of technical skills to be evaluated.

    Args:
        jd_analysis: The structured JD analysis dictionary.
        interview_plan: The structured interview plan dictionary.
    Returns:
        A tuple of TechnicalSkillSpec objects representing the skills to evaluate.

    Raises:
        PermanentEvaluationError: If no technical skills could be inferred.
    """
    priorities = _priority_map(jd_analysis)
    specs: list[TechnicalSkillSpec] = []
    seen: set[str] = set()

    planned_sections = [
        section
        for section in _json_list(interview_plan.get("sections"))
        if isinstance(section, dict)
        and str(section.get("section_name") or "").casefold()
        not in {"self_intro", "behavioural_cultural"}
        and str(section.get("skill") or section.get("section_name") or "").strip()
    ]
    planned_minutes = [
        _allocated_minutes(section.get("allocated_mins"))
        for section in planned_sections
    ]
    average_topic_mins = (
        sum(planned_minutes) / len(planned_minutes) if planned_minutes else 0.0
    )

    for section in planned_sections:
        name = str(section.get("skill") or section.get("section_name") or "").strip()
        section_name = str(section.get("section_name") or "").casefold()
        if not name or section_name in {"self_intro", "behavioural_cultural"}:
            continue
        key = _normalize_skill_key(name)
        if not key or key in seen:
            continue
        expected_signals = tuple(
            str(item).strip()
            for item in _json_list(section.get("expected_signals"))
            if str(item).strip()
        )
        matched_priority = _matched_priority_for_skill(name, priorities)
        allocated_mins = _allocated_minutes(section.get("allocated_mins"))
        specs.append(
            TechnicalSkillSpec(
                name=name,
                priority_score=(
                    matched_priority
                    if matched_priority is not None
                    else _time_aware_manual_priority(
                        allocated_mins,
                        average_topic_mins,
                    )
                ),
                expected_signals=expected_signals,
            )
        )
        seen.add(key)

    if not specs:
        for item in _json_list(jd_analysis.get("skills")):
            if not isinstance(item, dict):
                continue
            name = str(item.get("skill") or item.get("name") or "").strip()
            key = _normalize_skill_key(name)
            if not name or key in seen:
                continue
            specs.append(
                TechnicalSkillSpec(
                    name=name,
                    priority_score=_priority_for_skill(name, priorities),
                    expected_signals=(),
                )
            )
            seen.add(key)
    if not specs:
        raise PermanentEvaluationError(
            "Evaluation requires at least one planned technical skill"
        )
    return tuple(specs)


def _canonicalize_question_skills(
    qa_pairs: list[QuestionAnswerPair],
    technical_skills: tuple[TechnicalSkillSpec, ...],
) -> tuple[list[QuestionAnswerPair], tuple[TechnicalSkillSpec, ...]]:
    specs_by_key = {_normalize_skill_key(item.name): item for item in technical_skills}
    counts = {item.name: 0 for item in technical_skills}
    canonical_pairs: list[QuestionAnswerPair] = []

    for pair in qa_pairs:
        value = pair.model_dump(mode="json")
        pair_key = _normalize_skill_key(pair.skill) if pair.skill else ""
        matched = specs_by_key.get(pair_key)
        if matched is None and pair_key:
            for spec_key, spec in specs_by_key.items():
                if min(len(pair_key), len(spec_key)) >= 3 and (
                    pair_key in spec_key or spec_key in pair_key
                ):
                    matched = spec
                    break
        if matched is not None:
            value["skill"] = matched.name
            if not value["expected_signals"]:
                value["expected_signals"] = list(matched.expected_signals)
            counts[matched.name] += 1
        canonical_pairs.append(QuestionAnswerPair.model_validate(value))

    counted_specs = tuple(
        TechnicalSkillSpec(
            name=item.name,
            priority_score=item.priority_score,
            expected_signals=item.expected_signals,
            questions_asked=counts[item.name],
        )
        for item in technical_skills
    )
    return canonical_pairs, counted_specs


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
