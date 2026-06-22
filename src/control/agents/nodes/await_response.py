"""Await-response node.

This project does not use LangGraph interrupt checkpoints yet. The node is an
explicit graph pause marker: each invocation ends here, and the WebSocket
submits the next candidate transcript through the orchestrator.
"""

from __future__ import annotations

import logging

from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)


async def await_response(state: InterviewState) -> dict:
    logger.info(
        "Awaiting candidate response: ca_id=%s turn=%s",
        state.get("candidate_assessment_id", "unknown"),
        state.get("turn_number", 0),
    )
    return {"next_node": "await_response"}
