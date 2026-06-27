"""Bridge between LiveKit voice turns and the LangGraph interview workflow."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langgraph.types import Command

from src.control.agents.graphs import get_graph
from src.data.repositories import (
    assessment_context_repository,
    interview_session_repository,
)
from src.utils.interview_graph import utc_now_iso
from src.utils.interview_turns import extract_bot_reply

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"COMPLETED", "TERMINATED", "DEACTIVATED", "EVALUATED"}


class LiveKitInterviewBridge:
    """Direct stateful adapter between LiveKit turns and LangGraph."""

    def __init__(self, *, candidate_assessment_id: str) -> None:
        self.candidate_assessment_id = candidate_assessment_id
        self._graph_instance: Any | None = None
        self.state: dict[str, Any] | None = None

    @property
    def _config(self) -> dict[str, dict[str, str]]:
        return {
            "configurable": {
                "thread_id": self.candidate_assessment_id,
            }
        }

    async def _graph(self) -> Any:
        if self._graph_instance is None:
            self._graph_instance = await get_graph()
        return self._graph_instance

    async def _snapshot(self) -> dict[str, Any] | None:
        graph = await self._graph()
        snapshot = await graph.aget_state(self._config)
        values = getattr(snapshot, "values", None) if snapshot else None
        return dict(values) if values else None

    async def _state_after(self, result: Any) -> dict[str, Any]:
        state = await self._snapshot()
        if state is not None:
            return state
        return dict(result) if isinstance(result, dict) else {}

    async def _start_graph(self) -> dict[str, Any]:
        graph = await self._graph()
        logger.info(
            "Starting interview graph from LiveKit bridge",
            extra={"candidate_assessment_id": self.candidate_assessment_id},
        )
        result = await graph.ainvoke(
            {
                "candidate_assessment_id": self.candidate_assessment_id,
                "thread_id": self.candidate_assessment_id,
                "interview_session_id": None,
            },
            config=self._config,
        )
        return await self._state_after(result)

    async def _resume_graph(
        self,
        candidate_event: dict[str, Any] | str,
    ) -> dict[str, Any]:
        graph = await self._graph()
        logger.info(
            "Sending LiveKit candidate turn directly to interview graph",
            extra={
                "candidate_assessment_id": self.candidate_assessment_id,
                "event_type": candidate_event.get("event_type")
                if isinstance(candidate_event, dict)
                else candidate_event,
            },
        )
        result = await graph.ainvoke(
            Command(resume=candidate_event),
            config=self._config,
        )
        return await self._state_after(result)

    async def _update_state(self, values: dict[str, Any]) -> dict[str, Any]:
        graph = await self._graph()
        await graph.aupdate_state(self._config, values)
        return await self._snapshot() or {}

    async def start_or_resume(self) -> str:
        """Start or resume the LangGraph interview and return new bot speech."""

        previous = self.state
        state = await self._snapshot()
        if state is None:
            state = await self._start_graph()
        elif not self._is_terminal(state):
            session_id = state.get("interview_session_id")
            if session_id:
                await interview_session_repository.mark_session_in_progress(
                    str(session_id)
                )

            replay_text = str(state.get("last_bot_text") or "").strip()
            state = await self._update_state(
                {
                    "session_status": "IN_PROGRESS",
                    "grace_period_expires_at": None,
                    "disconnected_at": None,
                    "reconnect_count": int(state.get("reconnect_count") or 0) + 1,
                    "resumed": True,
                    "bot_reply_text": replay_text,
                    "bot_reply_type": "replay" if replay_text else "",
                }
            )

        self.state = state
        return extract_bot_reply(previous, state)

    @staticmethod
    def _is_terminal(state: dict[str, Any]) -> bool:
        return bool(
            state.get("closing_done")
            or str(state.get("session_status") or "").upper() in TERMINAL_STATUSES
        )

    async def start_timer(self) -> bool:
        """Start the durable time budget when the interviewer first speaks."""

        state = self.state or await self._snapshot()
        if not state or state.get("started_at"):
            self.state = state
            return False

        started_at = utc_now_iso()
        self.state = await self._update_state(
            {
                "started_at": started_at,
                "section_started_at": state.get("section_started_at") or started_at,
            },
        )
        await assessment_context_repository.mark_candidate_timer_started(
            self.candidate_assessment_id
        )
        return True

    async def submit_candidate_turn(
        self,
        text: str,
        *,
        stt_confidence: float | None = None,
        duration_ms: int | None = None,
    ) -> str:
        """Submit one LiveKit-confirmed candidate turn to LangGraph."""

        previous = self.state
        state = await self._snapshot()
        if state is None:
            state = await self._start_graph()
        if self._is_terminal(state):
            self.state = state
            return extract_bot_reply(previous, state)

        if text in {"__SILENCE__", "__TIME_UP__"}:
            candidate_event: dict[str, Any] | str = text
        else:
            candidate_event = {
                "event_type": "candidate_answer",
                "text": text,
                "stt_confidence": stt_confidence,
                "duration_ms": duration_ms,
                "ended_at": datetime.now(UTC).isoformat(),
                "eot_source": "livekit_turn_detector_v1",
                "raw_metadata": {},
            }

        state = await self._resume_graph(candidate_event)
        self.state = state
        return extract_bot_reply(previous, state)

    async def submit_silence(self) -> str:
        """Submit a graph silence sentinel after LiveKit user-away timeout."""

        return await self.submit_candidate_turn("__SILENCE__")

    async def close(self, *, terminated: bool = False) -> str:
        """Close the graph session and return any closing speech."""

        previous = self.state
        state = await self._snapshot()
        if state is None or self._is_terminal(state):
            return ""

        state = await self._resume_graph(
            {
                "event_type": (
                    "candidate_disconnect" if terminated else "timer_expired"
                ),
                "text": "",
            }
        )
        self.state = state
        return extract_bot_reply(previous, state)
