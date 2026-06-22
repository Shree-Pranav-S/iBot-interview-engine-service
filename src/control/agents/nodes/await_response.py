"""Await-response node.

This is the graph's suspension point. LangGraph pauses here until the
WebSocket layer resumes the run with ``Command(resume=transcript)``.
"""

from __future__ import annotations

import logging

from langgraph.types import interrupt

from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)


async def await_response(state: InterviewState) -> dict:
    ca_id = state.get("candidate_assessment_id", "unknown")
    turn_number = int(state.get("turn_number") or 0)
    logger.info(
        "Awaiting candidate response: ca_id=%s turn=%s",
        ca_id,
        turn_number,
    )

    transcript = interrupt(
        {
            "reason": "awaiting_candidate_response",
            "candidate_assessment_id": ca_id,
            "turn_number": turn_number,
            "current_question": state.get("current_question_text")
            or state.get("current_question")
            or "",
        }
    )

    if isinstance(transcript, dict):
        text = str(transcript.get("text") or transcript.get("transcript") or "")
        confidence = transcript.get("confidence")
    else:
        text = str(transcript or "")
        confidence = None

    logger.info(
        "Candidate transcript resumed graph: ca_id=%s turn=%s len=%s",
        ca_id,
        turn_number,
        len(text),
    )

    return {
        "candidate_raw_text": text,
        "candidate_stt_confidence": confidence,
        "next_node": "classify_response",
    }
