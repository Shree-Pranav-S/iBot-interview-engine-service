"""Simple next-action decision node for the low-latency workflow."""

from __future__ import annotations

from src.control.agents.nodes.context_utils import (
    behavioural_cultural_requirement_met,
    behavioural_cultural_section_index,
    is_behavioural_cultural_section,
    is_self_intro_section,
)
from src.control.agents.state import InterviewState

EARLY_COMPLETION_BUFFER_SECS = 8


def _has_next_section(state: InterviewState) -> bool:
    return (
        int(state.get("current_section_index") or 0)
        < len(state.get("section_order") or []) - 1
    )


def decide_next_action(state: InterviewState) -> dict[str, object]:
    if (
        state.get("close_after_behavioural_cultural")
        and is_behavioural_cultural_section(state)
        and behavioural_cultural_requirement_met(state)
    ):
        return {
            "next_action": "complete",
            "should_close": True,
            "closing_reason": "time_expired_after_behavioural_cultural",
        }

    if is_self_intro_section(state) and _has_next_section(state):
        return {
            "next_action": "section_transition",
            "next_section_index": int(state.get("current_section_index") or 0) + 1,
            "should_close": False,
        }

    if state.get("force_transition_due_to_overrun") and _has_next_section(state):
        return {
            "next_action": "section_transition",
            "next_section_index": int(state.get("current_section_index") or 0) + 1,
            "should_close": False,
        }

    remaining_secs = int(state.get("remaining_secs") or 0)
    behavioural_index = behavioural_cultural_section_index(state)
    if (
        0 < remaining_secs <= 20
        and behavioural_index is not None
        and not is_behavioural_cultural_section(state)
        and not behavioural_cultural_requirement_met(state)
    ):
        return {
            "next_action": "section_transition",
            "next_section_index": behavioural_index,
            "force_behavioural_cultural_due_to_time": True,
            "force_transition_due_to_overrun": True,
            "close_after_behavioural_cultural": True,
            "should_close": False,
        }

    if remaining_secs > EARLY_COMPLETION_BUFFER_SECS:
        return {"next_action": "next_question", "should_close": False}

    return {
        "next_action": "complete",
        "should_close": True,
        "closing_reason": state.get("closing_reason") or "time_expired",
    }
