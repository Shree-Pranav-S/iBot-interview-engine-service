"""Build immutable transcript and Q&A context for holistic evaluation."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from src.core.exceptions.evaluation import PermanentEvaluationError
from src.schemas.evaluation_llm import (
    EvaluationCandidateContext,
    EvaluationInput,
    QuestionAnswerPair,
)

EVALUATION_SCHEMA_VERSION = "holistic-evaluation-v2"


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


def normalize_skill_key(value: Any) -> str:
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
            priorities[normalize_skill_key(name)] = _score(
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
    key = normalize_skill_key(skill)
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


def build_question_answer_pairs(
    transcript: list[dict[str, Any]],
) -> list[QuestionAnswerPair]:
    """Pair every scored interviewer question with its candidate responses."""

    pair_data: list[dict[str, Any]] = []
    pair_indexes: dict[str, int] = {}
    active_index: int | None = None

    for turn_index, turn in enumerate(transcript):
        speaker = str(turn.get("speaker") or "").casefold()
        if speaker == "bot" and _is_scored_question(turn):
            metadata = _json_object(turn.get("metadata"))
            raw_id = _question_id(turn)
            question_id = raw_id or f"legacy-question-{turn_index + 1}"
            # Duplicate identifiers should represent the same question (for
            # example, a rephrase). A genuinely new question without a stable
            # id gets the deterministic legacy id above.
            existing_index = pair_indexes.get(question_id)
            if existing_index is not None:
                active_index = existing_index
                continue

            section = str(
                turn.get("current_section")
                or turn.get("section")
                or metadata.get("current_section")
                or metadata.get("section")
                or "unknown"
            ).strip()
            skill_value = (
                turn.get("current_skill")
                or turn.get("skill")
                or metadata.get("current_skill")
                or metadata.get("skill")
            )
            skill = str(skill_value).strip() if skill_value else None
            difficulty = str(
                turn.get("question_difficulty")
                or turn.get("difficulty")
                or metadata.get("question_difficulty")
                or metadata.get("difficulty")
                or "unknown"
            ).strip()
            question_text = str(
                turn.get("question_text")
                or metadata.get("question_text")
                or turn.get("text")
                or ""
            ).strip()
            if not question_text:
                continue
            pair_data.append(
                {
                    "question_id": question_id,
                    "section": section or "unknown",
                    "skill": skill,
                    "difficulty": difficulty or "unknown",
                    "question_text": question_text,
                    "bot_turn_id": (
                        str(turn["turn_id"]) if turn.get("turn_id") else None
                    ),
                    "answer_turn_ids": [],
                    "answers": [],
                    "response_types": [],
                    "live_evaluations": [],
                    "expected_signals": _string_list(
                        turn.get("expected_signals") or metadata.get("expected_signals")
                    ),
                    "answered": False,
                }
            )
            active_index = len(pair_data) - 1
            pair_indexes[question_id] = active_index
            continue

        if speaker != "candidate":
            continue
        candidate_question_id = _question_id(turn)
        target_index = (
            pair_indexes.get(candidate_question_id, active_index)
            if candidate_question_id
            else active_index
        )
        if target_index is None:
            continue
        pair = pair_data[target_index]
        answer_text = str(turn.get("text") or "").strip()
        response_type = _response_type(turn) or "unknown"
        if answer_text:
            pair["answers"].append(answer_text)
        pair["response_types"].append(response_type)
        if turn.get("turn_id"):
            pair["answer_turn_ids"].append(str(turn["turn_id"]))
        live_evaluation = _live_evaluation(turn)
        if live_evaluation:
            pair["live_evaluations"].append(live_evaluation)
        if _meaningful_answer(answer_text, response_type):
            pair["answered"] = True

    return [QuestionAnswerPair.model_validate(item) for item in pair_data]


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

    for section in _json_list(interview_plan.get("sections")):
        if not isinstance(section, dict):
            continue
        name = str(section.get("skill") or section.get("section_name") or "").strip()
        section_name = str(section.get("section_name") or "").casefold()
        if not name or section_name in {"self_intro", "behavioural_cultural"}:
            continue
        key = normalize_skill_key(name)
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
            )
        )
        seen.add(key)

    if not specs:
        for item in _json_list(jd_analysis.get("skills")):
            if not isinstance(item, dict):
                continue
            name = str(item.get("skill") or item.get("name") or "").strip()
            key = normalize_skill_key(name)
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
    specs_by_key = {normalize_skill_key(item.name): item for item in technical_skills}
    counts = {item.name: 0 for item in technical_skills}
    canonical_pairs: list[QuestionAnswerPair] = []

    for pair in qa_pairs:
        value = pair.model_dump(mode="json")
        pair_key = normalize_skill_key(pair.skill) if pair.skill else ""
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


def build_evaluation_context(
    source: dict[str, Any],
) -> EvaluationContextBundle:
    """
    Validate source data, extract skills, calculate properties, and build
    the complete immutable JSON context bundle for evaluation.

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
    qa_pairs = build_question_answer_pairs(transcript)
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
    )
    qa_pairs, technical_skills = _canonicalize_question_skills(
        qa_pairs,
        technical_skills,
    )
    if not qa_pairs:
        raise PermanentEvaluationError(
            "Cannot evaluate an interview without any scored questions"
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
        qa_pairs=qa_pairs,
        violations=violations,
    )
    return EvaluationContextBundle(
        evaluation_input=evaluation_input,
        technical_skills=technical_skills,
    )
