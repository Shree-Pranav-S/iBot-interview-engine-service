"""Closing node for the interview graph."""

from __future__ import annotations

from typing import Any

from src.control.agents.nodes.context_utils import _next_bot_turn_number
from src.control.agents.state import InterviewState
from src.utils.interview_graph import deterministic_turn_id, utc_now_iso


def _closing_text(state: InterviewState) -> str:
    reason = state.get("closing_reason")
    if reason == "total_overrun" or state.get("force_close_due_to_overrun"):
        return "I have enough information from this interview. Thank you for completing it; the next step will be handled after this session."
    return "Thank you for completing the interview. I have captured your responses, and the next step will be handled after this session."


def generate_closing_message(state: InterviewState) -> dict[str, Any]:
    if state.get("closing_done"):
        return {
            "pending_bot_turn": None,
            "should_close": True,
            "next_action": "complete",
            "next_node": "final_evaluation",
        }

    session_id = str(state["interview_session_id"])
    turn_number = _next_bot_turn_number(state)
    text = _closing_text(state)
    turn = {
        "turn_id": deterministic_turn_id(session_id, turn_number, "bot"),
        "turn_number": turn_number,
        "speaker": "bot",
        "tone": "professional",
        "text": text,
        "section": state.get("current_section") or "closing",
        "skill": state.get("current_skill"),
        "difficulty": state.get("current_difficulty") or "medium",
        "question_id": state.get("current_question_id"),
        "timestamp": utc_now_iso(),
        "metadata": {"message_type": "closing"},
    }
    return {
        "pending_bot_turn": turn,
        "last_bot_text": text,
        "bot_reply_type": "closing",
        "should_close": True,
        "closing_done": True,
        "next_action": "complete",
    }
