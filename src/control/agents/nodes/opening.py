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
        f"Hi {candidate_name}, welcome to this interview with {company_name} for {role_name}. "
        "I will ask one question at a time. Please answer naturally, and we will keep moving through the interview.",
        f"Hello {candidate_name}. Thanks for joining this interview with {company_name} for {role_name}. "
        "We'll go through the questions one by one. Feel free to answer naturally as we progress.",
        f"Hi {candidate_name}, glad to have you here for the {company_name} interview for {role_name}. "
        "I'll be guiding you through a series of questions, one at a time. Just respond naturally, and we'll move through each section together.",
        f"Welcome, {candidate_name}. Thank you for taking the time to interview with {company_name} for {role_name}. "
        "I will present the questions one at a time. Please answer them naturally, and we will step through the process.",
        f"Hello {candidate_name}, welcome to this interview with {company_name} for {role_name}. "
        "I will ask one question at a time. Go ahead and answer naturally, and we will move along through each part.",
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
