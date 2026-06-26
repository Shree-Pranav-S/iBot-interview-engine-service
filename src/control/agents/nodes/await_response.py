"""Candidate response interrupt node."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from src.control.agents.state import InterviewState
from src.utils.interview_graph import deterministic_turn_id, utc_now_iso


def _from_text(value: str) -> dict[str, Any]:
    if value == "__SILENCE__":
        return {
            "event_type": "silence_timeout",
            "text": "",
            "silence_duration_ms": 10000,
        }
    if value == "__TIME_UP__":
        return {"event_type": "timer_expired", "text": ""}
    return {
        "event_type": "candidate_answer",
        "text": " ".join(value.split()),
        "ended_at": utc_now_iso(),
        "eot_source": "text_input",
        "raw_metadata": {},
    }


def await_candidate_response_interrupt(state: InterviewState) -> dict[str, Any]:
    raw = interrupt(
        {
            "event": "await_candidate_response",
            "candidate_assessment_id": state.get("candidate_assessment_id"),
            "question_id": state.get("current_question_id"),
            "question_text": state.get("current_question_text"),
        }
    )

    if isinstance(raw, str):
        event = _from_text(raw)
    elif isinstance(raw, dict):
        event = dict(raw)
    else:
        event = {"event_type": "candidate_answer", "text": ""}

    event_type = str(event.get("event_type") or "candidate_answer")
    response_type = {
        "candidate_answer": "answer",
        "candidate_clarification": "clarification_question",
        "silence_timeout": "silence",
        "candidate_disconnect": "disconnect",
        "technical_issue": "technical_issue",
        "candidate_interruption": "interruption",
        "candidate_skip": "skip",
        "timer_expired": "timer_expired",
    }.get(event_type, "answer")
    event["event_type"] = event_type
    event["response_type"] = response_type
    event.setdefault("text", "")
    event.setdefault("ended_at", utc_now_iso())
    turn_number = int(state.get("turn_number") or 0) + 1
    session_id = str(state["interview_session_id"])
    candidate_turn = {
        "turn_id": deterministic_turn_id(session_id, turn_number, "candidate"),
        "turn_number": turn_number,
        "speaker": "candidate",
        "tone": "silence" if response_type == "silence" else "neutral",
        "text": str(event.get("text") or ""),
        "section": state.get("current_section") or "general",
        "skill": state.get("current_skill"),
        "question_id": state.get("current_question_id"),
        "timestamp": event.get("ended_at") or utc_now_iso(),
        "metadata": {
            "response_type": response_type,
            "stt_confidence": event.get("stt_confidence"),
            "duration_ms": event.get("duration_ms"),
            "silence_duration_ms": event.get("silence_duration_ms"),
            "eot_source": event.get("eot_source"),
            "raw_metadata": event.get("raw_metadata") or {},
        },
    }

    return {
        "candidate_event": raw,
        "normalized_candidate_event": event,
        "last_candidate_event": event,
        "last_response_type": response_type,
        "pending_candidate_turn": candidate_turn,
        "bot_reply_text": "",
        "bot_reply_type": "",
        "next_node": None,
    }
