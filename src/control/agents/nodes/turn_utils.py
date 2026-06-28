"""Build durable JSON transcript records for graph turns."""

from __future__ import annotations

from typing import Any

from src.control.agents.state import InterviewState
from src.utils.interview_graph import deterministic_turn_id, utc_now_iso


def build_bot_turn(
    state: InterviewState,
    *,
    text: str,
    question_type: str,
    question_text: str | None = None,
    acknowledgement: str | None = None,
    topic: str | None = None,
) -> dict[str, Any]:
    """Build one bot transcript item for the current interaction number."""

    session_id = str(state["interview_session_id"])
    turn_number = int(state.get("turn_number") or 1)
    return {
        "turn_id": deterministic_turn_id(session_id, turn_number, "bot"),
        "turn_number": turn_number,
        "speaker": "bot",
        "text": " ".join(text.split()),
        "question_difficulty": state.get("current_question_difficulty"),
        "response_type": None,
        "question_type": question_type,
        "current_skill": state.get("current_technical_skill"),
        "current_section": state.get("current_section"),
        "question_id": state.get("current_question_id"),
        "timestamp": utc_now_iso(),
        "metadata": {
            "question_type": question_type,
            "question_text": question_text,
            "acknowledgement": acknowledgement,
            "expected_signals": list(state.get("current_expected_signals") or []),
            "topic": topic,
            "inferred_difficulty": state.get("inferred_difficulty"),
            "transition_reason": state.get("transition_reason"),
            "closing_reason": state.get("closing_reason"),
            "barge_in_triggered": bool(state.get("barge_in_triggered")),
            "elapsed_secs": state.get("elapsed_secs"),
            "section_elapsed_secs": state.get("current_section_elapsed_secs"),
        },
    }


def build_candidate_turn(
    state: InterviewState,
    event: dict[str, Any],
) -> dict[str, Any]:
    """Build a candidate item; classification enriches it before persistence."""

    session_id = str(state["interview_session_id"])
    turn_number = int(state.get("turn_number") or 1)
    return {
        "turn_id": deterministic_turn_id(session_id, turn_number, "candidate"),
        "turn_number": turn_number,
        "speaker": "candidate",
        "text": str(event.get("text") or ""),
        "question_difficulty": None,
        "response_type": None,
        "question_type": None,
        "current_skill": state.get("current_technical_skill"),
        "current_section": state.get("current_section"),
        "question_id": state.get("current_question_id"),
        "timestamp": event.get("received_at") or utc_now_iso(),
        "metadata": {
            "stt_confidence": event.get("stt_confidence"),
            "duration_ms": event.get("duration_ms"),
            "silence_duration_ms": event.get("silence_duration_ms"),
            "question_text": state.get("current_question_text"),
            "question_difficulty": state.get("current_question_difficulty"),
            "elapsed_secs": event.get("elapsed_secs"),
        },
    }
