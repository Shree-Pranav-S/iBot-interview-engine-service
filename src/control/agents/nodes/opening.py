"""Opening message node."""

from __future__ import annotations

import random
from typing import Any

from src.control.agents.state import InterviewState
from src.utils.interview_graph import deterministic_turn_id, utc_now_iso


def generate_opening_message(state: InterviewState) -> dict[str, Any]:
    session_id = str(state["interview_session_id"])
    turn_number = int(state.get("turn_number") or 0) + 1
    candidate_name = state.get("candidate_name") or "there"
    role_name = state.get("role_name") or "the role"
    company_name = state.get("company_name") or "the company"

    opening_texts = [
        f"Hi {candidate_name}, welcome to your {company_name} interview for {role_name}.",
        f"Hello {candidate_name}, thanks for joining the {company_name} interview for {role_name}.",
        f"Welcome, {candidate_name}. This is your {company_name} interview for {role_name}.",
        f"Hi {candidate_name}, glad to have you here for the {role_name} interview with {company_name}.",
        f"Hello {candidate_name}, welcome to this live interview with {company_name}.",
    ]
    text = random.choice(opening_texts)

    turn = {
        "turn_id": deterministic_turn_id(session_id, turn_number, "bot"),
        "turn_number": turn_number,
        "speaker": "bot",
        "tone": "professional",
        "text": text,
        "section": state.get("current_section") or "opening",
        "skill": None,
        "difficulty": state.get("current_difficulty") or "medium",
        "question_id": None,
        "timestamp": utc_now_iso(),
        "metadata": {"message_type": "opening"},
    }
    return {
        "pending_bot_turn": turn,
        "last_bot_text": text,
        "bot_reply_type": "opening",
        "next_node": "persist_interview_turn",
    }
