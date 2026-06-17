"""
trigger_evaluation_node — Stub for holistic evaluation trigger.

Full implementation in Phase 2: enqueues Celery task for
comprehensive post-interview analysis.
"""

from __future__ import annotations

import logging

from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)


async def trigger_evaluation_node(state: InterviewState) -> dict:
    """Enqueue holistic evaluation task (stub — Phase 2)."""
    logger.info(
        "Holistic evaluation triggered: assessment=%s total_turns=%d",
        state.get("candidate_assessment_id", "unknown"),
        state.get("turn_number", 0),
    )

    # In Phase 2: Celery task enqueue here
    return {
        "session_status": "completed",
    }
