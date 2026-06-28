"""Static opening delivery node."""

from __future__ import annotations

from typing import Any

from src.control.agents.nodes.turn_utils import build_bot_turn
from src.control.agents.state import InterviewState
from src.control.agents.templates import choose_template


def deliver_opening(state: InterviewState) -> dict[str, Any]:
    """Select one of ten openings without starting the interview timer."""

    text = choose_template(
        "opening",
        candidate_name=state.get("candidate_name") or "Candidate",
        company_name=state.get("company_name") or "the company",
    )
    question_text = str(state["current_question_text"])
    pending_bot_turn = build_bot_turn(
        state,
        text=text,
        question_type="opening",
        question_text=question_text,
        topic="professional background",
    )
    asked_questions = list(state.get("asked_questions") or [])
    if not any(
        item.get("question_id") == state.get("current_question_id")
        for item in asked_questions
    ):
        asked_questions.append(
            {
                "question_id": state.get("current_question_id"),
                "question_text": question_text,
                "difficulty": None,
                "skill": None,
                "section": state.get("current_section"),
                "topic": "professional background",
            }
        )
    return {
        "bot_reply_text": text,
        "bot_reply_type": "opening",
        "pending_bot_turn": pending_bot_turn,
        "asked_questions": asked_questions,
        "next_action": "await_candidate_response",
        "timer_started": False,
        "timer_started_at": None,
    }
