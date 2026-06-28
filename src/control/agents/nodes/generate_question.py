"""Strict technical and behavioural/cultural question generation."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.control.agents.nodes.llm_helpers import generate_with_schema
from src.control.agents.nodes.question_strategy import (
    determine_question_difficulty,
    target_section,
)
from src.control.agents.nodes.turn_utils import build_bot_turn
from src.control.agents.prompts import (
    BEHAVIOURAL_QUESTION_GENERATION_SYSTEM_PROMPT,
    TECHNICAL_QUESTION_GENERATION_SYSTEM_PROMPT,
)
from src.control.agents.state import Difficulty, InterviewState
from src.control.agents.templates import (
    choose_template_avoiding,
    template_variants,
)
from src.schemas.prompts import (
    BehaviouralQuestionGenerationResponse,
    TechnicalQuestionGenerationResponse,
)

logger = logging.getLogger(__name__)


def _normalized_question(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9+#.]+", value.casefold()))


def _question_tokens(value: str) -> set[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "can",
        "could",
        "describe",
        "do",
        "explain",
        "how",
        "in",
        "is",
        "of",
        "the",
        "to",
        "what",
        "when",
        "with",
        "would",
        "you",
        "your",
    }
    return {
        token
        for token in _normalized_question(value).split()
        if token not in stop_words
    }


def _questions_for_skill(
    state: InterviewState,
    skill: str,
) -> list[dict[str, Any]]:
    return [
        item
        for item in (state.get("asked_questions") or [])
        if str(item.get("skill") or "").casefold() == skill.casefold()
    ]


def _behavioural_questions(state: InterviewState) -> list[dict[str, Any]]:
    return [
        item
        for item in (state.get("asked_questions") or [])
        if item.get("section") == "behavioural_cultural"
    ]


def _recent_acknowledgements(state: InterviewState) -> list[str]:
    return [
        str(item.get("acknowledgement") or "").strip()
        for item in (state.get("asked_questions") or [])
        if str(item.get("acknowledgement") or "").strip()
    ][-6:]


def _validate_acknowledgement(
    acknowledgement: str,
    recent_acknowledgements: list[str],
) -> None:
    pass
    pass


def _candidate_evaluation_context(state: InterviewState) -> dict[str, Any] | None:
    eval_dict = state.get("latest_evaluation")
    if (
        state.get("last_response_type") == "answer"
        and state.get("last_response_substantial")
        and isinstance(eval_dict, dict)
    ):
        return dict(eval_dict)
    return None


def _technical_messages(
    state: InterviewState,
    *,
    skill: str,
    expected_signals: list[str],
    target_difficulty: Difficulty,
    probe_deeper: bool,
) -> list[dict[str, str]]:
    asked = _questions_for_skill(state, skill)
    suppress_previous = bool(state.get("suppress_previous_context_for_next_question"))
    context = {
        "current_technical_skill": skill,
        "expected_signals": expected_signals,
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
        "question_sequence_number": len(state.get("asked_questions") or []) + 1,
        "recent_acknowledgements": _recent_acknowledgements(state),
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
        "schema": TechnicalQuestionGenerationResponse.model_json_schema(),
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
    asked = _behavioural_questions(state)
    suppress_previous = bool(state.get("suppress_previous_context_for_next_question"))
    context = {
        "expected_signals": expected_signals,
        "previous_question": (
            None if suppress_previous else state.get("current_question_text")
        ),
        "previous_candidate_response": (
            None if suppress_previous else state.get("previous_candidate_response")
        ),
        "question_variation_seed": state.get("question_variation_seed"),
        "question_sequence_number": len(state.get("asked_questions") or []) + 1,
        "recent_acknowledgements": _recent_acknowledgements(state),
        "questions_already_asked": [item.get("question_text") for item in asked],
        "signals_already_used": [
            item.get("topic") for item in asked if item.get("topic")
        ],
        "schema": BehaviouralQuestionGenerationResponse.model_json_schema(),
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


def _validate_unique_question(
    question_text: str,
    asked_questions: list[dict[str, Any]],
    *,
    allow_related_probe: bool = False,
) -> None:
    normalized = _normalized_question(question_text)
    new_tokens = _question_tokens(question_text)
    for item in asked_questions:
        prior_text = str(item.get("question_text") or "")
        if normalized == _normalized_question(prior_text):
            raise ValueError("question generator repeated a previous question")
        if allow_related_probe:
            continue
        prior_tokens = _question_tokens(prior_text)
        union = new_tokens | prior_tokens
        if union and len(new_tokens & prior_tokens) / len(union) >= 0.72:
            raise ValueError(
                "question generator lightly paraphrased a previous question"
            )


async def _generate_technical(
    state: InterviewState,
    *,
    skill: str,
    expected_signals: list[str],
    target_difficulty: Difficulty,
    probe_deeper: bool,
) -> TechnicalQuestionGenerationResponse:
    messages = _technical_messages(
        state,
        skill=skill,
        expected_signals=expected_signals,
        target_difficulty=target_difficulty,
        probe_deeper=probe_deeper,
    )
    asked = _questions_for_skill(state, skill)
    used_topics = {
        str(topic).strip().casefold()
        for topic in (state.get("used_topics_by_skill") or {}).get(
            skill.casefold(),
            [],
        )
    }
    recent_acknowledgements = _recent_acknowledgements(state)
    for attempt in range(2):
        try:
            result = await generate_with_schema(
                messages,
                TechnicalQuestionGenerationResponse,
            )
            if result.difficulty != target_difficulty:
                raise ValueError("question generator changed deterministic difficulty")
            if result.probe_deeper is not probe_deeper:
                raise ValueError("question generator changed deterministic probe flag")
            _validate_acknowledgement(
                result.acknowledgement,
                recent_acknowledgements,
            )
            _validate_unique_question(
                result.question_text,
                asked,
                allow_related_probe=probe_deeper,
            )
            if not probe_deeper and result.topic.strip().casefold() in used_topics:
                raise ValueError("question generator reused a previous topic")
            return result
        except Exception as exc:
            logger.warning(
                "Technical question generation attempt %s failed: %s",
                attempt + 1,
                exc,
            )
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "The previous output violated the schema, repeated prior "
                        "content or acknowledgement wording, or changed deterministic "
                        "controls. Return a new valid JSON object with a distinct "
                        "acknowledgement opening, the exact target difficulty, and "
                        "the exact probe_deeper value."
                    ),
                },
            ]

    signal = expected_signals[0] if expected_signals else f"practical {skill} use"
    candidates = template_variants(
        "technical_question_fallback",
        skill=skill,
        signal=signal,
    )
    difficulty_indices = {
        "easy": (0, 1, 2),
        "medium": (2, 3, 4, 5, 6, 7),
        "hard": (3, 5, 6, 7, 8, 9),
    }[target_difficulty]
    for index in difficulty_indices:
        question = candidates[index]
        try:
            _validate_unique_question(
                question,
                asked,
                allow_related_probe=probe_deeper,
            )
        except ValueError:
            continue
        return TechnicalQuestionGenerationResponse(
            acknowledgement=choose_template_avoiding(
                "question_generation_fallback_acknowledgement",
                recent=recent_acknowledgements,
            ),
            question_text=question,
            difficulty=target_difficulty,
            probe_deeper=probe_deeper,
            topic=f"{signal} application",
        )
    raise RuntimeError("Unable to generate a unique technical fallback question")


async def _generate_behavioural(
    state: InterviewState,
    *,
    expected_signals: list[str],
) -> BehaviouralQuestionGenerationResponse:
    messages = _behavioural_messages(
        state,
        expected_signals=expected_signals,
    )
    asked = _behavioural_questions(state)
    recent_acknowledgements = _recent_acknowledgements(state)
    for attempt in range(2):
        try:
            result = await generate_with_schema(
                messages,
                BehaviouralQuestionGenerationResponse,
            )
            _validate_acknowledgement(
                result.acknowledgement,
                recent_acknowledgements,
            )
            _validate_unique_question(result.question_text, asked)
            return result
        except Exception as exc:
            logger.warning(
                "Behavioural question generation attempt %s failed: %s",
                attempt + 1,
                exc,
            )
            messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "The previous output was invalid or repeated an earlier "
                        "question or acknowledgement style. Return a different "
                        "valid JSON object with a new acknowledgement opening."
                    ),
                },
            ]

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
            _validate_unique_question(question, asked)
        except ValueError:
            continue
        return BehaviouralQuestionGenerationResponse(
            acknowledgement=choose_template_avoiding(
                "question_generation_fallback_acknowledgement",
                recent=recent_acknowledgements,
            ),
            question_text=question,
            signal_focus=signal,
        )
    raise RuntimeError("Unable to generate a unique behavioural fallback question")


async def generate_next_question(state: InterviewState) -> dict[str, Any]:
    """Generate the next question for the selected timed section."""

    section_index, section = target_section(state)
    section_kind = str(section.get("section_kind") or "technical")
    section_name = str(section.get("section_name") or "technical")
    expected_signals = list(section.get("expected_signals") or [])
    entering_new_section = (
        section_index != int(state.get("current_section_index") or 0)
        or state.get("pending_section_index") is not None
    )
    generation_state: InterviewState = {
        **state,
        "suppress_previous_context_for_next_question": (
            bool(state.get("suppress_previous_context_for_next_question"))
            or entering_new_section
        ),
    }
    previous_question = state.get("current_question_text")
    previous_difficulty = state.get("current_question_difficulty")
    previous_evaluation = _candidate_evaluation_context(state)

    if section_kind == "technical":
        skill = str(section.get("skill") or section_name)
        difficulty, probe_deeper = determine_question_difficulty(
            state,
            skill=skill,
            entering_new_section=entering_new_section,
        )
        generated = await _generate_technical(
            generation_state,
            skill=skill,
            expected_signals=expected_signals,
            target_difficulty=difficulty,
            probe_deeper=probe_deeper,
        )
        acknowledgement = (
            str(state.get("response_preface_text") or "").strip()
            or generated.acknowledgement
        )
        question_text = generated.question_text
        topic = generated.topic
        current_skill: str | None = skill
        current_difficulty: Difficulty | None = difficulty
    else:
        generated_behavioural = await _generate_behavioural(
            generation_state,
            expected_signals=expected_signals,
        )
        acknowledgement = (
            str(state.get("response_preface_text") or "").strip()
            or generated_behavioural.acknowledgement
        )
        question_text = generated_behavioural.question_text
        topic = generated_behavioural.signal_focus
        current_skill = None
        current_difficulty = None
        probe_deeper = False

    question_id = (
        f"{state['interview_session_id']}:question:"
        f"{len(state.get('asked_questions') or []) + 1}"
    )
    spoken_text = f"{acknowledgement} {question_text}".strip()
    section_budgets = state.get("section_budgets_secs") or {}
    section_budget = int(section_budgets.get(str(section_index), 60))
    section_started_elapsed = (
        int(state.get("elapsed_secs") or 0)
        if entering_new_section
        else int(state.get("current_section_started_elapsed_secs") or 0)
    )
    section_elapsed = max(
        0,
        int(state.get("elapsed_secs") or 0) - section_started_elapsed,
    )
    runtime_sections = list(state.get("runtime_sections") or [])
    future_reserved = sum(
        max(0, int(section_budgets.get(str(index)) or 0))
        for index, future_section in enumerate(runtime_sections)
        if index > section_index
        and future_section.get("section_kind")
        in {
            "technical",
            "behavioural_cultural",
        }
    )
    total_duration = max(1, int(state.get("total_duration_secs") or 1))
    protected_deadline = (
        total_duration - future_reserved if future_reserved > 0 else total_duration
    )
    section_deadline = max(
        0,
        min(section_started_elapsed + section_budget, protected_deadline),
    )
    section_remaining = max(
        0,
        min(
            section_budget - section_elapsed,
            section_deadline - int(state.get("elapsed_secs") or 0),
        ),
    )
    next_state: InterviewState = {
        **state,
        "current_section": section_name,
        "current_section_index": section_index,
        "current_section_kind": section_kind,  # type: ignore[typeddict-item]
        "current_technical_skill": current_skill,
        "current_expected_signals": expected_signals,
        "current_question_id": question_id,
        "current_question_text": question_text,
        "last_rephrased_question": None,
        "current_question_difficulty": current_difficulty,
        "is_self_introduction": False,
        "current_section_budget_secs": section_budget,
        "current_section_started_elapsed_secs": section_started_elapsed,
        "current_section_elapsed_secs": section_elapsed,
        "current_section_remaining_secs": section_remaining,
        "reserved_future_section_secs": future_reserved,
        "current_section_transition_deadline_elapsed_secs": (section_deadline),
    }
    pending_bot_turn = build_bot_turn(
        next_state,
        text=spoken_text,
        question_type="new_question",
        question_text=question_text,
        acknowledgement=acknowledgement,
        topic=topic,
    )
    question_record = {
        "question_id": question_id,
        "question_text": question_text,
        "difficulty": current_difficulty,
        "skill": current_skill,
        "section": section_name,
        "topic": topic,
        "acknowledgement": acknowledgement,
    }
    asked_questions = [
        *list(state.get("asked_questions") or []),
        question_record,
    ]
    used_topics = {
        key: list(value)
        for key, value in (state.get("used_topics_by_skill") or {}).items()
    }
    if current_skill:
        used_topics.setdefault(current_skill.casefold(), []).append(topic)
    return {
        "previous_question_text": previous_question,
        "previous_question_difficulty": previous_difficulty,
        "previous_evaluation": previous_evaluation,
        "current_section": section_name,
        "current_section_index": section_index,
        "current_section_kind": section_kind,
        "current_technical_skill": current_skill,
        "current_expected_signals": expected_signals,
        "current_question_id": question_id,
        "current_question_text": question_text,
        "last_rephrased_question": None,
        "current_question_difficulty": current_difficulty,
        "is_self_introduction": False,
        "probe_deeper": probe_deeper,
        "asked_questions": asked_questions,
        "used_topics_by_skill": used_topics,
        "current_section_budget_secs": section_budget,
        "current_section_started_elapsed_secs": section_started_elapsed,
        "current_section_elapsed_secs": section_elapsed,
        "current_section_remaining_secs": max(
            0,
            section_remaining,
        ),
        "reserved_future_section_secs": future_reserved,
        "current_section_transition_deadline_elapsed_secs": section_deadline,
        "pending_section_index": None,
        "suppress_previous_context_for_next_question": False,
        "transition_reason": None,
        "barge_in_triggered": False,
        "bot_reply_text": spoken_text,
        "bot_reply_type": "new_question",
        "pending_bot_turn": pending_bot_turn,
        "response_preface_text": None,
        "skip_attempts_for_current_question": 0,
        "self_intro_elaboration_requested": False,
        "should_advance_question": False,
        "phase_complete": False,
        "latest_evaluation": None,
        "next_action": "await_candidate_response",
        "silence_stage": "none",
    }
