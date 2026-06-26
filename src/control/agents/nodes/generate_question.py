"""Minimal-context interview question generation."""

from __future__ import annotations

from typing import Any

from src.control.agents.nodes.context_utils import (
    NON_TECH_SECTIONS,
    _next_bot_turn_number,
    is_behavioural_cultural_section_name,
)
from src.control.agents.nodes.llm_helpers import (
    clean_text,
    compact_json,
    generate_json,
    model_to_dict,
)
from src.control.agents.prompts import (
    BEHAVIOURAL_QUESTION_SYSTEM_PROMPT,
    QUESTION_GENERATION_SYSTEM_PROMPT,
)
from src.control.agents.state import InterviewState
from src.schemas.prompts import QuestionGenerationResponse
from src.utils.interview_graph import (
    clean_section_name,
    deterministic_turn_id,
    utc_now_iso,
)

SELF_INTRO_QUESTION = (
    "To start, could you briefly introduce yourself and highlight the experience "
    "most relevant to this role?"
)


def _plan_section(state: InterviewState, section_index: int) -> dict[str, Any]:
    runtime_sections = state.get("runtime_sections") or []
    if 0 <= section_index < len(runtime_sections):
        item = runtime_sections[section_index]
        if isinstance(item, dict):
            return item
    return {}


def _count_section_questions(
    asked_questions: list[dict[str, Any]],
    section: str,
) -> int:
    return sum(
        1
        for item in asked_questions
        if item.get("section") == section and not item.get("is_followup")
    )


def _previous_questions(
    state: InterviewState,
    *,
    section: str,
    skill: str | None,
) -> list[str]:
    questions: list[str] = []
    for item in state.get("asked_questions") or []:
        if not isinstance(item, dict):
            continue
        same_skill = skill and str(item.get("skill") or "").lower() == skill.lower()
        same_section = item.get("section") == section
        if same_skill or same_section:
            text = str(item.get("question_text") or "").strip()
            if text:
                questions.append(text)
    return questions[-10:]


def _latest_candidate_answer(state: InterviewState) -> str:
    event = state.get("normalized_candidate_event") or {}
    text = str(event.get("text") or "").strip()
    if text:
        return text
    for turn in reversed(state.get("recent_turns") or []):
        if str(turn.get("speaker") or "").lower() == "candidate":
            return str(turn.get("text") or "").strip()
    return ""


def _question_messages(
    state: InterviewState,
    *,
    section: str,
    skill: str | None,
    difficulty: str,
) -> list[dict[str, str]]:
    evaluation = state.get("latest_evaluation") or {}
    context = {
        "current_section": section,
        "current_skill": skill,
        "target_difficulty": difficulty,
        "previous_response": _latest_candidate_answer(state),
        "previous_response_strength": evaluation.get("strength"),
        "previous_response_difficulty": evaluation.get("recommended_difficulty")
        or state.get("current_difficulty"),
        "previous_questions": _previous_questions(
            state,
            section=section,
            skill=skill,
        ),
    }
    system_prompt = (
        BEHAVIOURAL_QUESTION_SYSTEM_PROMPT
        if is_behavioural_cultural_section_name(section)
        else QUESTION_GENERATION_SYSTEM_PROMPT
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Return JSON matching this schema. Ask exactly one fresh question. "
                f"{compact_json(QuestionGenerationResponse.model_json_schema(), max_chars=1800)}. "
                f"Context: {compact_json(context, max_chars=2600)}"
            ),
        },
    ]


def _static_self_intro_question(
    state: InterviewState,
    *,
    section: str,
    question_id: str,
    turn_number: int,
) -> dict[str, Any]:
    session_id = str(state["interview_session_id"])
    turn = {
        "turn_id": deterministic_turn_id(session_id, turn_number, "bot"),
        "turn_number": turn_number,
        "speaker": "bot",
        "tone": "professional",
        "text": SELF_INTRO_QUESTION,
        "section": section,
        "skill": None,
        "difficulty": "easy",
        "question_id": question_id,
        "timestamp": utc_now_iso(),
        "metadata": {
            "message_type": "question",
            "expected_signals": [],
            "max_answer_secs": 60,
        },
    }
    return {
        "question_text": SELF_INTRO_QUESTION,
        "difficulty": "easy",
        "expected_signals": [],
        "turn": turn,
    }


async def generate_question(state: InterviewState) -> dict[str, Any]:
    session_id = str(state["interview_session_id"])
    turn_number = _next_bot_turn_number(state)
    asked_questions = list(state.get("asked_questions") or [])
    section_order = state.get("section_order") or [
        state.get("current_section") or "general"
    ]
    section_index = min(
        max(0, int(state.get("current_section_index") or 0)),
        len(section_order) - 1,
    )
    section = clean_section_name(section_order[section_index])
    plan_section = _plan_section(state, section_index)
    skill = plan_section.get("skill") or (
        None if section in NON_TECH_SECTIONS else section.replace("_", " ").title()
    )
    question_number = _count_section_questions(asked_questions, section) + 1
    question_id = f"q_{section}_{question_number:03d}"
    target_difficulty = str(state.get("current_difficulty") or "medium").lower()

    if section == "self_intro":
        generated = _static_self_intro_question(
            state,
            section=section,
            question_id=question_id,
            turn_number=turn_number,
        )
        question_text = generated["question_text"]
        difficulty = generated["difficulty"]
        expected_signals = generated["expected_signals"]
        turn = generated["turn"]
    else:
        response = model_to_dict(
            await generate_json(
                _question_messages(
                    state,
                    section=section,
                    skill=skill,
                    difficulty=target_difficulty,
                ),
                QuestionGenerationResponse,
            )
        )
        question_text = clean_text(response["question_text"], max_chars=300)
        difficulty = str(response.get("difficulty") or target_difficulty).lower()
        expected_signals = response.get("expected_signals") or []
        section_remaining_secs = int(state.get("current_section_remaining_secs") or 0)
        max_answer_secs = max(10, min(90, section_remaining_secs or 60))
        turn = {
            "turn_id": deterministic_turn_id(session_id, turn_number, "bot"),
            "turn_number": turn_number,
            "speaker": "bot",
            "tone": "professional",
            "text": question_text,
            "section": section,
            "skill": skill,
            "difficulty": difficulty,
            "question_id": question_id,
            "timestamp": utc_now_iso(),
            "metadata": {
                "message_type": "question",
                "expected_signals": expected_signals,
                "max_answer_secs": max_answer_secs,
                "generation_rationale": response.get("rationale"),
            },
        }

    question_record = {
        "question_id": question_id,
        "question_text": question_text,
        "section": section,
        "skill": skill,
        "difficulty": difficulty,
        "is_followup": False,
        "expected_signals": expected_signals,
    }
    return {
        "current_section": section,
        "current_section_index": section_index,
        "current_section_name": section,
        "current_skill": skill,
        "current_question_id": question_id,
        "current_question_text": question_text,
        "current_difficulty": difficulty,
        "asked_questions": [*asked_questions, question_record],
        "clarification_count_for_current_question": 0,
        "silence_count_for_current_question": 0,
        "non_answer_count_for_current_question": 0,
        "skip_count_for_current_question": 0,
        "think_silence_count": 0,
        "awaiting_think_confirmation": False,
        "think_extension_active": False,
        "pending_bot_turn": turn,
        "last_bot_text": question_text,
        "bot_reply_type": "question",
        "next_action": None,
        "next_node": "persist_interview_turn",
    }
