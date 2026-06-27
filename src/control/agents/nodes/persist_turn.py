"""Common persistence node for interview turns and violations."""

from __future__ import annotations

import logging
from typing import Any

from src.control.agents.state import InterviewState
from src.data.repositories import interview_session_repository
from src.utils.interview_graph import append_recent_turn

logger = logging.getLogger(__name__)


AWAITING_BOT_MESSAGE_TYPES = {
    "question",
    "clarification",
    "nudge",
    "rephrase",
    "redirect",
    "think_offer",
    "think_wait",
}


def _elapsed_secs(state: InterviewState) -> int:
    return max(0, int(state.get("elapsed_secs") or 0))


def _next_node_for_bot_turn(message_type: str) -> str:
    if message_type in AWAITING_BOT_MESSAGE_TYPES:
        return "await_candidate_response_interrupt"
    if message_type == "closing":
        return "final_evaluation"
    return "check_time_budget"


async def _persist_transcript_turn(
    state: InterviewState,
    turn: dict[str, Any],
) -> bool:
    appended = await interview_session_repository.append_transcript_turn(
        str(state["interview_session_id"]),
        turn,
        total_elapsed_secs=_elapsed_secs(state),
    )
    logger.info(
        "persisted interview transcript turn",
        extra={
            "candidate_assessment_id": state.get("candidate_assessment_id"),
            "turn_id": turn.get("turn_id"),
            "speaker": turn.get("speaker"),
            "appended": appended,
        },
    )
    return appended


async def _append_violation(
    state: InterviewState,
    violation: dict[str, Any],
) -> bool:
    appended = await interview_session_repository.append_violation(
        str(state["interview_session_id"]),
        violation,
    )
    logger.info(
        "persisted interview violation",
        extra={
            "candidate_assessment_id": state.get("candidate_assessment_id"),
            "violation_id": violation.get("violation_id"),
            "violation_type": violation.get("violation_type"),
            "appended": appended,
        },
    )
    return appended


async def persist_interview_turn(state: InterviewState) -> dict[str, Any]:
    """Persist pending candidate, bot, and violation artifacts in one node."""

    recent_turns = list(state.get("recent_turns") or [])
    turn_number = int(state.get("turn_number") or 0)
    updates: dict[str, Any] = {
        "pending_candidate_turn": None,
        "pending_bot_turn": None,
        "violation_to_persist": None,
    }

    candidate_turn = state.get("pending_candidate_turn")
    if candidate_turn:
        await _persist_transcript_turn(state, candidate_turn)
        recent_turns = append_recent_turn(recent_turns, candidate_turn)
        turn_number = max(turn_number, int(candidate_turn.get("turn_number") or 0))
        updates.update(
            {
                "last_candidate_event": state.get("normalized_candidate_event"),
                "last_response_type": state.get("last_response_type")
                or (state.get("normalized_candidate_event") or {}).get("response_type"),
            }
        )

    bot_turn = state.get("pending_bot_turn")
    bot_message_type = ""
    if bot_turn:
        await _persist_transcript_turn(state, bot_turn)
        recent_turns = append_recent_turn(recent_turns, bot_turn)
        turn_number = max(turn_number, int(bot_turn.get("turn_number") or 0))

        metadata = (
            bot_turn.get("metadata", {})
            if isinstance(bot_turn.get("metadata"), dict)
            else {}
        )
        bot_message_type = str(metadata.get("message_type") or "bot")
        previous_reply = str(state.get("bot_reply_text") or "").strip()
        text = str(bot_turn.get("text") or "").strip()
        updates.update(
            {
                "last_bot_text": text or state.get("last_bot_text"),
                "bot_reply_text": " ".join(
                    item for item in [previous_reply, text] if item
                ).strip(),
                "bot_reply_type": bot_message_type,
            }
        )

    violation = state.get("violation_to_persist")
    if violation:
        await _append_violation(state, violation)

    next_node = state.get("next_node")
    if bot_message_type:
        next_node = _next_node_for_bot_turn(bot_message_type)

    updates.update(
        {
            "recent_turns": recent_turns[-8:],
            "turn_number": turn_number,
            "next_node": next_node,
        }
    )
    return updates
