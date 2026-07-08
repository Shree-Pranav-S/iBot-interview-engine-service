"""Build immutable transcript and Q&A context for holistic evaluation."""

from __future__ import annotations

from typing import Any

from src.core.exceptions.evaluation import PermanentEvaluationError
from src.schemas.evaluation_llm import (
    EvaluationCandidateContext,
    EvaluationInput,
    QuestionAnswerPair,
)
from src.schemas.internal_evaluation import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationContextBundle,
)
from src.utils.evaluation_context import (
    _canonicalize_question_skills,
    _is_scored_question,
    _json_list,
    _json_object,
    _live_evaluation,
    _meaningful_answer,
    _normalize_skill_key,
    _question_id,
    _response_type,
    _string_list,
    _technical_skill_specs,
    _transcript_hash,
)


def normalize_skill_key(value: Any) -> str:
    """
    Normalize a skill name into a consistent lowercase string for matching.

    Args:
        value: The raw skill name.

    Returns:
        A normalized string containing only alphanumeric characters and allowed symbols (+, #, .).
    """
    return _normalize_skill_key(value)


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
