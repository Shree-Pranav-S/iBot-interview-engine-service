"""Helpers for technical and behavioural interview question generation."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.control.agents.key_routing import active_turn_key_slot
from src.control.agents.prompts import (
    BEHAVIOURAL_QUESTION_GENERATION_SYSTEM_PROMPT,
    TECHNICAL_QUESTION_GENERATION_SYSTEM_PROMPT,
)
from src.control.agents.state import Difficulty, InterviewState
from src.control.agents.templates import (
    choose_template_avoiding,
    template_variants,
)
from src.control.agents.utils.question_diversity import (
    framing_hint,
    recent_acknowledgements,
    recent_question_stems,
    validate_generated_question,
    validate_unique_question,
)
from src.control.agents.utils.question_strategy import (
    behavioural_questions,
    question_brief_for_skill,
    questions_for_skill,
)
from src.core.exceptions import (
    QuestionGenerationFailedException,
    QuestionGeneratorContractException,
    QuestionValidationException,
)
from src.core.services import llm_service
from src.schemas.prompts import (
    BehaviouralQuestionGenerationResponse,
    TechnicalQuestionGenerationResponse,
)

logger = logging.getLogger(__name__)

_FALLBACK_CONCEPTS = (
    "memory lifecycle",
    "concurrency hazard",
    "schema migration risk",
    "connection pooling",
    "error propagation",
    "caching consistency",
    "input validation gap",
    "deployment rollback",
    "observability signal",
    "resource leak",
)


def _fallback_concept_for_skill(state: InterviewState, skill: str) -> str:
    """Pick an unused fallback concept for this skill (last-resort path only)."""

    asked = questions_for_skill(state, skill)
    used_topics = {
        str(topic).casefold()
        for topic in (state.get("used_topics_by_skill") or {}).get(skill.casefold(), [])
    }
    for concept in _FALLBACK_CONCEPTS:
        if concept.casefold() not in used_topics:
            return concept
    return _FALLBACK_CONCEPTS[len(asked) % len(_FALLBACK_CONCEPTS)]


def _candidate_evaluation_context(state: InterviewState) -> dict[str, Any] | None:
    """
    Extract the evaluation of the immediately preceding answer, if it was substantial.
    This helps the generator tailor follow-up questions to the candidate's performance.

    Args:
        state: The current interview state.

    Returns:
        The evaluation dictionary, or None if the previous turn was not an answer.
    """
    eval_dict = state.get("latest_evaluation")
    if (
        state.get("last_response_type") == "answer"
        and state.get("last_response_substantial")
        and isinstance(eval_dict, dict)
    ):
        strength = str(eval_dict.get("strength") or "").strip()
        return {"strength": strength} if strength else None
    return None


def _technical_messages(
    state: InterviewState,
    *,
    skill: str,
    target_difficulty: Difficulty,
    probe_deeper: bool,
) -> list[dict[str, str]]:
    """
    Construct the LLM prompt for generating a new technical question.

    Args:
        state: The current interview state.
        skill: The specific technical skill being assessed.
        target_difficulty: The required difficulty level.
        probe_deeper: Whether this should be a follow-up to the last question.

    Returns:
        A list of chat messages for the LLM.
    """
    asked = questions_for_skill(state, skill)
    suppress_previous = bool(state.get("suppress_previous_context_for_next_question"))
    sequence_number = len(state.get("asked_questions") or []) + 1
    seed = str(state.get("question_variation_seed") or "")
    context = {
        "current_technical_skill": skill,
        "jd_question_brief": question_brief_for_skill(state, skill),
        "inferred_difficulty": state.get("inferred_difficulty"),
        "target_question_difficulty": target_difficulty,
        "previous_question": (
            None if suppress_previous else state.get("current_question_text")
        ),
        "previous_question_difficulty": (
            None if suppress_previous else state.get("current_question_difficulty")
        ),
        "previous_candidate_response": (
            None if suppress_previous else state.get("previous_candidate_response")
        ),
        "previous_evaluation": (
            None if suppress_previous else _candidate_evaluation_context(state)
        ),
        "probe_deeper": probe_deeper,
        "question_variation_seed": state.get("question_variation_seed"),
        "question_sequence_number": sequence_number,
        "question_framing_hint": framing_hint(seed, sequence_number),
        "recent_question_stems": recent_question_stems(state),
        "recent_acknowledgements": recent_acknowledgements(state, limit=6),
        "questions_already_asked_for_skill": [
            item.get("question_text") for item in asked
        ],
        "topics_already_used_for_skill": list(
            (state.get("used_topics_by_skill") or {}).get(
                skill.casefold(),
                [],
            )
        ),
        "resume_context": state.get("resume_context") or {},
    }
    return [
        {
            "role": "system",
            "content": TECHNICAL_QUESTION_GENERATION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, default=str),
        },
    ]


def _behavioural_messages(
    state: InterviewState,
    *,
    expected_signals: list[str],
) -> list[dict[str, str]]:
    """
    Construct the LLM prompt for generating a behavioural or cultural question.

    Args:
        state: The current interview state.
        expected_signals: Values/traits to look for.

    Returns:
        A list of chat messages for the LLM.
    """
    asked = behavioural_questions(state)
    suppress_previous = bool(state.get("suppress_previous_context_for_next_question"))
    sequence_number = len(state.get("asked_questions") or []) + 1
    seed = str(state.get("question_variation_seed") or "")
    context = {
        "expected_signals": expected_signals,
        "previous_question": (
            None if suppress_previous else state.get("current_question_text")
        ),
        "previous_candidate_response": (
            None if suppress_previous else state.get("previous_candidate_response")
        ),
        "question_variation_seed": state.get("question_variation_seed"),
        "question_sequence_number": sequence_number,
        "question_framing_hint": framing_hint(seed, sequence_number),
        "recent_question_stems": recent_question_stems(state),
        "recent_acknowledgements": recent_acknowledgements(state, limit=6),
        "questions_already_asked": [item.get("question_text") for item in asked],
        "signals_already_used": [
            item.get("topic") for item in asked if item.get("topic")
        ],
    }
    return [
        {
            "role": "system",
            "content": BEHAVIOURAL_QUESTION_GENERATION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, default=str),
        },
    ]


async def _generate_technical(
    state: InterviewState,
    *,
    skill: str,
    target_difficulty: Difficulty,
    probe_deeper: bool,
) -> TechnicalQuestionGenerationResponse:
    """
    Generate and strictly validate a technical question.
    Retries once if the model output fails validation. Falls back to a set of
    pre-written template questions if it repeatedly fails.

    Args:
        state: The interview state.
        skill: The skill to assess.
        target_difficulty: Required difficulty.
        probe_deeper: True if this is a follow-up.

    Returns:
        A validated technical question generation response.
    """
    messages = _technical_messages(
        state,
        skill=skill,
        target_difficulty=target_difficulty,
        probe_deeper=probe_deeper,
    )
    asked = questions_for_skill(state, skill)
    recent_acknowledgement_list = recent_acknowledgements(state, limit=6)
    try:
        result = await llm_service.generate(
            messages,
            TechnicalQuestionGenerationResponse,
            key_slot=active_turn_key_slot(state),
        )
        if result.difficulty != target_difficulty:
            raise QuestionGeneratorContractException(
                "question generator changed deterministic difficulty"
            )
        if result.probe_deeper is not probe_deeper:
            raise QuestionGeneratorContractException(
                "question generator changed deterministic probe flag"
            )
        validate_generated_question(
            question_text=result.question_text,
            topic=result.topic,
            skill=skill,
            asked_questions=asked,
            used_topics=list(
                (state.get("used_topics_by_skill") or {}).get(skill.casefold(), [])
            ),
            allow_related_probe=probe_deeper,
        )
        logger.info(
            "Technical question generated by LLM",
            extra={"question_source": "llm", "skill": skill},
        )
        return result
    except Exception as exc:
        logger.warning(
            "Technical question generation failed; using local fallback: %s",
            exc,
        )

    fallback_concept = _fallback_concept_for_skill(state, skill)
    logger.warning(
        "Using per-skill fallback question",
        extra={
            "question_source": "fallback",
            "skill": skill,
            "concept": fallback_concept,
        },
    )
    fallback_topic = fallback_concept
    candidates = template_variants(
        "technical_question_fallback",
        skill=skill,
        concept=fallback_concept,
        signal=fallback_topic,
    )
    difficulty_indices = {
        "easy": (0, 1, 2),
        "medium": (2, 3, 4, 5, 6, 7),
        "hard": (3, 5, 6, 7, 8, 9),
    }[target_difficulty]
    fallback_index = difficulty_indices[len(asked) % len(difficulty_indices)]
    return TechnicalQuestionGenerationResponse(
        acknowledgement=choose_template_avoiding(
            "question_generation_fallback_acknowledgement",
            recent=recent_acknowledgement_list,
        ),
        question_text=candidates[fallback_index],
        difficulty=target_difficulty,
        probe_deeper=probe_deeper,
        topic=fallback_topic,
    )


async def _generate_behavioural(
    state: InterviewState,
    *,
    expected_signals: list[str],
) -> BehaviouralQuestionGenerationResponse:
    """
    Generate and validate a behavioural/cultural question.
    Falls back locally if the model output is invalid.

    Args:
        state: The interview state.
        expected_signals: Behaviours/traits to probe.

    Returns:
        A validated behavioural question generation response.
    """
    messages = _behavioural_messages(
        state,
        expected_signals=expected_signals,
    )
    asked = behavioural_questions(state)
    recent_acknowledgement_list = recent_acknowledgements(state, limit=6)
    try:
        result = await llm_service.generate(
            messages,
            BehaviouralQuestionGenerationResponse,
            key_slot=active_turn_key_slot(state),
        )
        validate_generated_question(
            question_text=result.question_text,
            topic=result.signal_focus,
            skill="",
            asked_questions=asked,
            used_topics=[
                str(item.get("topic") or "") for item in asked if item.get("topic")
            ],
            allow_related_probe=False,
        )
        return result
    except Exception as exc:
        logger.warning(
            "Behavioural question generation failed; using local fallback: %s",
            exc,
        )

    signal = (
        expected_signals[len(asked) % len(expected_signals)]
        if expected_signals
        else "adaptability"
    )
    for question in template_variants(
        "behavioural_question_fallback",
        signal=signal,
    ):
        try:
            validate_unique_question(question, asked)
        except QuestionValidationException:
            continue
        return BehaviouralQuestionGenerationResponse(
            acknowledgement=choose_template_avoiding(
                "question_generation_fallback_acknowledgement",
                recent=recent_acknowledgement_list,
            ),
            question_text=question,
            signal_focus=signal,
        )
    raise QuestionGenerationFailedException(
        "Unable to generate a unique behavioural fallback question"
    )
