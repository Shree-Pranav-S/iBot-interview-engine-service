"""Static section transition node for the interview graph."""

from __future__ import annotations

from typing import Any

from src.control.agents.nodes.context_utils import _next_bot_turn_number
from src.control.agents.state import InterviewState
from src.utils.interview_graph import (
    clean_section_name,
    deterministic_turn_id,
    utc_now_iso,
)

URGENT_BEHAVIOURAL_TRANSITIONS = (
    "Sorry but we need to move on to the behavioural questions due to lack of time,",
)
OVERRUN_TRANSITIONS = (
    "Thanks for that. We need to move on to {display} now because we are short "
    "on time.",
    "Thanks. I need to move us to {display} now so we can stay within time.",
    "I appreciate that answer. We are short on time, so let us move to {display}.",
    "Thanks for sharing that. I am going to shift us to {display} now.",
    "Got it. We need to continue with {display} now because of the remaining time.",
    "Thank you. I will move us to {display} now so we can cover the next area.",
    "Thanks. Time is tight, so we need to go to {display} now.",
    "I have captured that. Let us move to {display} now due to time.",
    "Thanks for the response. We need to switch to {display} now.",
    "Understood. We will move to {display} now to keep the interview on track.",
)
STANDARD_TRANSITIONS = (
    "Thanks for that. Let's move to {display}.",
    "Thank you. We will move to {display} now.",
    "Got it. Let's continue with {display}.",
    "Thanks. I will move us into {display}.",
    "I have captured that. Let's go to {display}.",
    "Thank you for the answer. The next section is {display}.",
    "Great, let's shift to {display}.",
    "Understood. We will continue with {display}.",
    "Thanks for sharing that. Let's move ahead to {display}.",
    "All right, let's continue into {display}.",
)


def _runtime_section(state: InterviewState, index: int) -> dict[str, Any]:
    sections = state.get("runtime_sections") or []
    if 0 <= index < len(sections):
        item = sections[index]
        if isinstance(item, dict):
            return item
    return {}


def _display_name(state: InterviewState, index: int, section: str) -> str:
    item = _runtime_section(state, index)
    return str(item.get("display_name") or section.replace("_", " ").title())


def _pick(options: tuple[str, ...], state: InterviewState, salt: str) -> str:
    basis = f"{state.get('current_question_id')}-{state.get('turn_number')}-{salt}"
    index = sum(ord(char) for char in basis) % len(options)
    return options[index]


def _transition_text(state: InterviewState, display: str) -> str:
    if state.get("force_behavioural_cultural_due_to_time"):
        return _pick(URGENT_BEHAVIOURAL_TRANSITIONS, state, "urgent-behavioural")
    if state.get("force_transition_due_to_overrun"):
        return _pick(OVERRUN_TRANSITIONS, state, "overrun").format(display=display)
    return _pick(STANDARD_TRANSITIONS, state, "standard").format(display=display)


async def generate_section_transition(state: InterviewState) -> dict[str, Any]:
    section_order = state.get("section_order") or []
    current_index = int(state.get("current_section_index") or 0)
    next_index = state.get("next_section_index")
    if next_index is None:
        next_index = current_index + 1
    next_index = int(next_index)

    if next_index >= len(section_order):
        return {
            "next_action": "complete",
            "should_close": True,
            "closing_reason": state.get("closing_reason") or "sections_complete",
        }

    session_id = str(state["interview_session_id"])
    turn_number = _next_bot_turn_number(state)
    next_section = clean_section_name(section_order[next_index])
    section_data = _runtime_section(state, next_index)
    display = _display_name(state, next_index, next_section)
    text = _transition_text(state, display)

    turn = {
        "turn_id": deterministic_turn_id(session_id, turn_number, "bot"),
        "turn_number": turn_number,
        "speaker": "bot",
        "tone": "professional",
        "text": text,
        "section": next_section,
        "skill": section_data.get("skill"),
        "difficulty": state.get("current_difficulty") or "medium",
        "question_id": None,
        "timestamp": utc_now_iso(),
        "metadata": {"message_type": "section_transition"},
    }
    return {
        "previous_section": state.get("current_section"),
        "current_section": next_section,
        "current_section_index": next_index,
        "current_section_name": next_section,
        "current_skill": section_data.get("skill"),
        "current_question_id": None,
        "current_question_text": None,
        "section_started_at": utc_now_iso(),
        "current_section_elapsed_secs": 0,
        "current_section_remaining_secs": int(
            (state.get("section_budgets") or {}).get(next_section) or 0
        ),
        "clarification_count_for_current_question": 0,
        "silence_count_for_current_question": 0,
        "non_answer_count_for_current_question": 0,
        "skip_count_for_current_question": 0,
        "think_silence_count": 0,
        "awaiting_think_confirmation": False,
        "think_extension_active": False,
        "next_section_index": None,
        "next_action": "next_question",
        "pending_bot_turn": turn,
        "last_bot_text": text,
        "bot_reply_type": "section_transition",
        "should_close": False,
        "force_transition_due_to_overrun": False,
        "force_behavioural_cultural_due_to_time": False,
    }
