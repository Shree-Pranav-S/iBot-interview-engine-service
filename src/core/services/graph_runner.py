"""Service wrapper for starting and resuming the interview LangGraph."""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import Command

from src.control.agents.graphs import get_graph

logger = logging.getLogger(__name__)


class InterviewGraphRunner:
    """Start and resume graph runs keyed by candidate_assessment_id."""

    def __init__(self, graph: Any | None = None) -> None:
        self.graph = graph

    async def _graph(self) -> Any:
        if self.graph is None:
            self.graph = await get_graph()
        return self.graph

    @staticmethod
    def _config(candidate_assessment_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": candidate_assessment_id}}

    async def _snapshot(
        self,
        candidate_assessment_id: str,
    ) -> dict[str, Any] | None:
        graph = await self._graph()
        snapshot = await graph.aget_state(self._config(candidate_assessment_id))
        values = getattr(snapshot, "values", None) if snapshot else None
        return dict(values) if values else None

    async def _state_after(
        self,
        candidate_assessment_id: str,
        result: Any,
    ) -> dict[str, Any]:
        state = await self._snapshot(candidate_assessment_id)
        if state is not None:
            return state
        return dict(result) if isinstance(result, dict) else {}

    async def start(self, candidate_assessment_id: str) -> dict[str, Any]:
        graph = await self._graph()
        initial_state = {
            "candidate_assessment_id": candidate_assessment_id,
            "thread_id": candidate_assessment_id,
            "interview_session_id": None,
        }
        logger.info(
            "starting interview graph",
            extra={"candidate_assessment_id": candidate_assessment_id},
        )
        result = await graph.ainvoke(
            initial_state,
            config=self._config(candidate_assessment_id),
        )
        return await self._state_after(candidate_assessment_id, result)

    async def resume(
        self,
        candidate_assessment_id: str,
        candidate_event: dict[str, Any] | str,
    ) -> dict[str, Any]:
        graph = await self._graph()
        logger.info(
            "resuming interview graph",
            extra={
                "candidate_assessment_id": candidate_assessment_id,
                "event_type": candidate_event.get("event_type")
                if isinstance(candidate_event, dict)
                else candidate_event,
            },
        )
        result = await graph.ainvoke(
            Command(resume=candidate_event),
            config=self._config(candidate_assessment_id),
        )
        return await self._state_after(candidate_assessment_id, result)

    async def update_state(
        self,
        candidate_assessment_id: str,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        graph = await self._graph()
        await graph.aupdate_state(self._config(candidate_assessment_id), values)
        state = await self._snapshot(candidate_assessment_id)
        return state or {}

    async def get_state(self, candidate_assessment_id: str) -> dict[str, Any] | None:
        return await self._snapshot(candidate_assessment_id)
