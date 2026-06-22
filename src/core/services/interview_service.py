"""InterviewOrchestrator bridges the WebSocket route to the LangGraph workflow."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from src.control.agents.graphs import close_graph, continue_graph, start_graph
from src.data.repositories import interview_workflow_repository as db

logger = logging.getLogger(__name__)

GRACE_PERIOD_SECS = 300


class InterviewOrchestrator:
    """Small project-compatible orchestrator for one live WebSocket session."""

    def __init__(self, *, candidate_assessment_id: str) -> None:
        self.candidate_assessment_id = candidate_assessment_id

    async def start_session(self) -> dict:
        logger.info(
            "Starting interview session: ca_id=%s", self.candidate_assessment_id
        )
        state = await start_graph(candidate_assessment_id=self.candidate_assessment_id)
        await db.persist_session_state(state)
        return state

    async def resume_session(self) -> dict | None:
        state = await start_graph(candidate_assessment_id=self.candidate_assessment_id)
        if not state.get("resumed"):
            return None
        await db.persist_session_state(state)
        return state

    async def submit_response(
        self,
        transcript: str,
        *,
        stt_confidence: float | None = None,
    ) -> dict:
        state = await continue_graph(
            self.candidate_assessment_id,
            candidate_text=transcript,
            stt_confidence=stt_confidence,
        )
        await db.persist_session_state(state)
        return state

    async def close_session(self, *, terminated: bool = False) -> dict | None:
        state = await self.get_current_state()
        if state is None:
            return None
        state = await close_graph(
            self.candidate_assessment_id,
            terminated=terminated,
            current_status=state.get("session_status", "COMPLETED"),
        )
        await db.persist_session_state(state)
        return state

    async def pause_session(self) -> None:
        state = await self.get_current_state()
        if state is not None and not state.get("closing_done"):
            now = datetime.now(UTC)
            state["session_status"] = "PAUSED"
            state["paused_at"] = now.isoformat()
            state["grace_period_expires_at"] = (
                now + timedelta(seconds=GRACE_PERIOD_SECS)
            ).isoformat()
            await db.persist_session_state(state)
        await db.mark_session_paused(self.candidate_assessment_id)

    async def get_current_state(self) -> dict | None:
        from src.control.agents.graphs.interview_graph import get_graph

        config = {"configurable": {"thread_id": self.candidate_assessment_id}}
        state_snap = await get_graph().aget_state(config)
        return state_snap.values if state_snap else None
