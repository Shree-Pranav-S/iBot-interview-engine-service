"""
await_response_node — Pauses graph execution awaiting candidate input.

Uses LangGraph's interrupt() to suspend the graph. The WebSocket handler
resumes with the candidate's STT transcript (or __SILENCE__ sentinel).
"""

from __future__ import annotations

import logging

from langgraph.types import interrupt

from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)


async def await_response_node(state: InterviewState) -> dict:
    """
    Interrupt graph execution and wait for candidate response.

    The WebSocket layer will resume the graph by calling
    graph.ainvoke(Command(resume=transcript), config=...) with the
    candidate's transcribed speech or "__SILENCE__" if the silence
    watchdog fires.
    """
    assessment_id = state.get("candidate_assessment_id", "unknown")
    turn = state.get("turn_number", 0)

    logger.info(
        "Awaiting candidate response: assessment=%s turn=%d",
        assessment_id,
        turn,
    )

    # interrupt() suspends the graph and waits for external input.
    # The value passed in is informational — the WebSocket handler
    # provides the actual transcript via Command(resume=...).
    transcript: str = interrupt(
        {
            "reason": "awaiting_candidate_response",
            "current_question": state.get("current_question_text", ""),
            "turn_number": turn,
        }
    )

    logger.info(
        "Candidate response received: assessment=%s turn=%d len=%d",
        assessment_id,
        turn,
        len(transcript),
    )

    return {
        "last_transcript": transcript,
        "turn_number": turn + 1,
    }
