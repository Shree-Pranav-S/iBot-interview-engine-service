"""Interrupt the graph until LiveKit supplies speech or a silence timeout."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from src.control.agents.key_routing import compute_turn_key_slot
from src.control.agents.nodes.turn_utils import build_candidate_turn
from src.control.agents.state import InterviewState
from src.utils.interview_graph import utc_now_iso


def _normalize_event(raw: Any) -> dict[str, Any]:
    """
    Ensure the incoming raw event from LiveKit is formatted as a consistent dictionary.

    Args:
        raw: The raw event which may be a string (like "__SILENCE__") or a dict.

    Returns:
        A normalized dictionary with at least an 'event_type' and 'text'.
    """
    if isinstance(raw, str):
        if raw == "__SILENCE__":
            return {
                "event_type": "silence_timeout",
                "text": "",
                "silence_duration_ms": 5000,
            }
        return {"event_type": "candidate_answer", "text": raw}

    if isinstance(raw, dict):
        return dict(raw)
    return {"event_type": "candidate_answer", "text": ""}


def await_candidate_response(state: InterviewState) -> dict[str, Any]:
    """
    Pause execution of the LangGraph state machine and yield control back to LiveKit.

    The node triggers a LangGraph `interrupt`. It stays suspended until the
    LiveKit agent invokes the graph again with a resume payload (like a speech
    transcript or a timeout event).

    Args:
        state: The current interview state.

    Returns:
        State updates containing the parsed candidate event and the next action route.
    """

    raw = interrupt(
        {
            "event": "await_candidate_response",
            "candidate_assessment_id": state.get("candidate_assessment_id"),
            "question_id": state.get("current_question_id"),
            "question_text": state.get("current_question_text"),
            "bot_reply_type": state.get("bot_reply_type"),
        }
    )
    event = _normalize_event(raw)
    event_type = str(event.get("event_type") or "candidate_answer")
    is_time_barge_in = event_type == "section_time_barge_in"
    is_silence = event_type == "silence_timeout"
    text = "" if is_silence else " ".join(str(event.get("text") or "").split())
    duration = event.get("duration_ms")

    try:
        duration_ms = max(0, int(duration)) if duration is not None else None
    except (TypeError, ValueError):
        duration_ms = None

    event.update(
        {
            "event_type": event_type,
            "text": text,
            "received_at": event.get("received_at") or utc_now_iso(),
            "duration_ms": duration_ms,
        }
    )
    pending_candidate_turn = (
        build_candidate_turn(state, event) if not is_time_barge_in or text else None
    )
    if is_time_barge_in and pending_candidate_turn:
        metadata = dict(pending_candidate_turn.get("metadata") or {})
        metadata["interrupted_by_section_time_barge_in"] = True
        pending_candidate_turn.update(
            {
                "response_type": "answer",
                "metadata": metadata,
            }
        )
    return {
        "candidate_event": event,
        "previous_candidate_response": text,
        "previous_response_duration_ms": duration_ms,
        "speculative_interviewer_result": event.get("speculative_interviewer_result"),
        "llm_key_slot": compute_turn_key_slot(state),
        "pending_candidate_turn": pending_candidate_turn,
        "bot_reply_text": "",
        "bot_reply_type": "",
        "next_action": (
            "force_section_time_barge_in" if is_time_barge_in else "interviewer_turn"
        ),
    }
