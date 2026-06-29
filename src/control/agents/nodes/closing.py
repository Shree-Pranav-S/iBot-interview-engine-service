"""Static, uninterruptible interview closing generation."""

from __future__ import annotations

from typing import Any

from src.control.agents.nodes.turn_utils import build_bot_turn
from src.control.agents.state import InterviewState
from src.control.agents.templates import choose_template


def generate_closing_message(state: InterviewState) -> dict[str, Any]:
    """
    Create one of ten stable closing messages for LiveKit playout and finalize graph state.

    This node transitions the interview to the CLOSING state. The resulting text is
    spoken by the bot, after which the LiveKit bridge triggers teardown.

    Args:
        state: The current interview state.

    Returns:
        State updates containing the closing bot turn and terminal routing keys.
    """

    text = choose_template(
        "closing",
        company_name=state.get("company_name") or "the company",
    )
    closing_state: InterviewState = {
        **state,
        "bot_reply_text": text,
        "bot_reply_type": "closing",
        "current_question_difficulty": None,
        "should_close": True,
        "closing_done": True,
        "session_status": "CLOSING",
    }
    pending_bot_turn = build_bot_turn(
        closing_state,
        text=text,
        question_type="closing",
    )
    return {
        "bot_reply_text": text,
        "bot_reply_type": "closing",
        "pending_bot_turn": pending_bot_turn,
        "should_close": True,
        "closing_done": True,
        "session_status": "CLOSING",
        "holistic_evaluation_status": "PENDING",
        "next_action": "end",
        "phase_complete": True,
    }
