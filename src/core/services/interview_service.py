"""InterviewOrchestrator bridges the WebSocket route to LangGraph.

The graph uses the canonical LangGraph pattern:
- start_session() invokes the graph until await_response interrupts;
- submit_response() resumes that interruption with Command(resume=...);
- checkpoints are keyed by candidate_assessment_id.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from langgraph.types import Command

from src.control.agents.graphs import get_graph
from src.data.repositories import interview_workflow_repository as db

logger = logging.getLogger(__name__)

GRACE_PERIOD_SECS = 300
TERMINAL_STATUSES = {"COMPLETED", "TERMINATED", "DEACTIVATED", "EVALUATED"}


class InterviewOrchestrator:
    """Coordinates one live WebSocket session with one LangGraph thread."""

    def __init__(self, *, candidate_assessment_id: str) -> None:
        self.candidate_assessment_id = candidate_assessment_id
        self._graph = get_graph()
        self._config: dict[str, dict[str, str]] = {
            "configurable": {"thread_id": candidate_assessment_id}
        }

    async def _snapshot_state(self) -> dict | None:
        snapshot = await self._graph.aget_state(self._config)
        values = getattr(snapshot, "values", None) if snapshot else None
        return dict(values) if values else None

    async def _state_after(self, result: Any) -> dict:
        state = await self._snapshot_state()
        if state is not None:
            return state
        return dict(result) if isinstance(result, dict) else {}

    async def _persist(self, state: dict | None) -> None:
        if state and state.get("session_id"):
            await db.persist_session_state(state)

    async def start_session(self) -> dict:
        """Start a fresh graph run and stop at the first await_response interrupt."""
        logger.info(
            "Starting interview session: ca_id=%s", self.candidate_assessment_id
        )
        result = await self._graph.ainvoke(
            {"candidate_assessment_id": self.candidate_assessment_id},
            config=self._config,
        )
        state = await self._state_after(result)
        await self._persist(state)
        return state

    async def resume_session(self) -> dict | None:
        """Return an existing interrupted graph state, if one is resumable."""
        state = await self._snapshot_state()
        if not state or not state.get("session_id"):
            return None

        if str(state.get("session_status") or "").upper() in TERMINAL_STATUSES:
            return None

        await db.mark_session_in_progress(self.candidate_assessment_id)
        state["session_status"] = "IN_PROGRESS"
        state["paused_at"] = None
        state["grace_period_expires_at"] = None
        await self._graph.aupdate_state(
            self._config,
            {
                "session_status": "IN_PROGRESS",
                "paused_at": None,
                "grace_period_expires_at": None,
            },
        )
        await self._persist(state)
        return state

    async def submit_response(
        self,
        transcript: str,
        *,
        stt_confidence: float | None = None,
    ) -> dict:
        """Resume the graph from await_response with the candidate transcript."""
        if await self._snapshot_state() is None:
            await self.start_session()

        resume_value: str | dict[str, object]
        if stt_confidence is None:
            resume_value = transcript
        else:
            resume_value = {"text": transcript, "confidence": stt_confidence}

        logger.info(
            "Resuming interview graph: ca_id=%s transcript_len=%d",
            self.candidate_assessment_id,
            len(transcript or ""),
        )
        result = await self._graph.ainvoke(
            Command(resume=resume_value),
            config=self._config,
        )
        state = await self._state_after(result)
        await self._persist(state)
        return state

    async def close_session(self, *, terminated: bool = False) -> dict | None:
        state = await self.get_current_state()
        if state is None:
            return None
        if state.get("closing_done"):
            return state

        sentinel = "__TIME_UP__"
        if terminated:
            await self._graph.aupdate_state(
                self._config,
                {"session_status": "TERMINATED", "should_close": True},
            )
        return await self.submit_response(sentinel)

    async def pause_session(self) -> None:
        state = await self.get_current_state()
        if state is not None and not state.get("closing_done"):
            now = datetime.now(UTC)
            state["session_status"] = "PAUSED"
            state["paused_at"] = now.isoformat()
            state["grace_period_expires_at"] = (
                now + timedelta(seconds=GRACE_PERIOD_SECS)
            ).isoformat()
            await self._graph.aupdate_state(
                self._config,
                {
                    "session_status": state["session_status"],
                    "paused_at": state["paused_at"],
                    "grace_period_expires_at": state["grace_period_expires_at"],
                },
            )
            await self._persist(state)
        await db.mark_session_paused(self.candidate_assessment_id)

    async def get_current_state(self) -> dict | None:
        return await self._snapshot_state()
