"""Stateful adapter between LiveKit voice events and the interview LangGraph."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from typing import Any, cast

from langgraph.types import Command

from src.control.agents.graphs import get_graph
from src.control.agents.nodes.persist_turn import (
    drain_background_persistence,
)
from src.control.agents.nodes.time_manager import (
    SECTION_BARGE_IN_GRACE_SECS,
    section_transition_deadline_elapsed,
)
from src.control.agents.state import InterviewState
from src.data.repositories import (
    assessment_context_repository,
    interview_session_repository,
)
from src.utils.interview_graph import utc_now_iso

logger = logging.getLogger(__name__)
TERMINAL_STATUSES = {"COMPLETED", "TERMINATED", "DEACTIVATED", "EVALUATED"}


class LiveKitInterviewBridge:
    """Resume one checkpointed graph thread from LiveKit-confirmed events."""

    def __init__(self, *, candidate_assessment_id: str) -> None:
        self.candidate_assessment_id = candidate_assessment_id
        self.state: dict[str, Any] | None = None
        self._config = {"configurable": {"thread_id": self.candidate_assessment_id}}
        self._timer_started_monotonic: float | None = None
        self._timer_started_at: datetime | None = None

    @staticmethod
    def _is_terminal(state: dict[str, Any]) -> bool:
        return str(state.get("session_status") or "").upper() in TERMINAL_STATUSES

    async def _checkpoint_state(self) -> dict[str, Any]:
        graph = await get_graph()
        snapshot = await graph.aget_state(self._config)
        values = getattr(snapshot, "values", None)
        return dict(values or {})

    async def start_or_resume(self) -> str:
        """Initialize context and opening, or restore the current spoken prompt."""

        graph = await get_graph()
        existing = await self._checkpoint_state()
        if existing:
            self.state = existing
            context = await assessment_context_repository.load_interview_context(
                self.candidate_assessment_id
            )
            started_at = context.get("interview_started_at")
            if isinstance(started_at, datetime):
                self._timer_started_at = started_at
                self.state["timer_started"] = True
                self.state["timer_started_at"] = started_at.isoformat()
            if self._is_terminal(existing):
                return ""
            return str(existing.get("bot_reply_text") or "").strip()

        result = await graph.ainvoke(
            {"candidate_assessment_id": self.candidate_assessment_id},
            config=self._config,
        )
        self.state = dict(result or {})
        return str(self.state.get("bot_reply_text") or "").strip()

    async def start_timer(self) -> bool:
        """Persist the timer only when LiveKit says first bot playout has begun."""

        if self.state and self.state.get("timer_started"):
            return False

        started_at = await assessment_context_repository.mark_candidate_timer_started(
            self.candidate_assessment_id
        )
        self._timer_started_monotonic = time.monotonic()
        self._timer_started_at = started_at
        session_id = str((self.state or {}).get("interview_session_id") or "")
        if session_id:
            await interview_session_repository.mark_session_in_progress(session_id)

        if self.state is None:
            self.state = {}
        self.state.update(
            {
                "timer_started": True,
                "timer_started_at": started_at.isoformat(),
            }
        )
        return True

    def elapsed_secs(self) -> int:
        """Return official elapsed interview time without starting the clock."""

        if self._timer_started_monotonic is not None:
            return max(
                0,
                int(time.monotonic() - self._timer_started_monotonic),
            )
        if self._timer_started_at is not None:
            started = self._timer_started_at
            if started.tzinfo is None:
                started = started.replace(tzinfo=UTC)
            return max(0, int((datetime.now(UTC) - started).total_seconds()))
        return max(0, int((self.state or {}).get("elapsed_secs") or 0))

    def seconds_until_section_barge_in(self) -> float | None:
        """Return delay to the grace deadline without stealing later sections."""

        state = self.state or {}
        if (
            not state
            or state.get("should_close")
            or state.get("closing_done")
            or self._is_terminal(state)
        ):
            return None
        budget = int(state.get("current_section_budget_secs") or 0)
        if budget <= 0:
            return None
        deadline_elapsed = section_transition_deadline_elapsed(
            cast(InterviewState, state),
            grace_secs=SECTION_BARGE_IN_GRACE_SECS,
        )
        return max(0.0, float(deadline_elapsed - self.elapsed_secs()))

    def _merge_result(self, result: Any) -> None:
        previous = self.state or {}
        timer_state = {
            "timer_started": bool(previous.get("timer_started")),
            "timer_started_at": previous.get("timer_started_at"),
        }
        self.state = {**dict(result or {}), **timer_state}

    async def submit_candidate_turn(
        self,
        text: str,
        *,
        stt_confidence: float | None = None,
        duration_ms: int | None = None,
    ) -> str:
        """Resume the pending graph interrupt with one final speech transcript."""

        if self.state is None:
            await self.start_or_resume()
        if not self.state or self._is_terminal(self.state):
            return ""
        if self.state.get("phase_complete"):
            return ""

        graph = await get_graph()
        event = {
            "event_type": "candidate_answer",
            "text": " ".join(text.split()),
            "stt_confidence": stt_confidence,
            "duration_ms": duration_ms,
            "elapsed_secs": self.elapsed_secs(),
            "received_at": utc_now_iso(),
        }
        result = await graph.ainvoke(
            Command(resume=event),
            config=self._config,
        )
        self._merge_result(result)
        return str(self.state.get("bot_reply_text") or "").strip()

    async def submit_silence(self) -> str:
        """Resume with LiveKit's five-second no-speech event; no LLM is called."""

        if self.state is None:
            await self.start_or_resume()
        if not self.state or self.state.get("phase_complete"):
            return ""

        graph = await get_graph()
        event = {
            "event_type": "silence_timeout",
            "text": "",
            "silence_duration_ms": 5000,
            "elapsed_secs": self.elapsed_secs(),
            "received_at": utc_now_iso(),
        }
        result = await graph.ainvoke(
            Command(resume=event),
            config=self._config,
        )
        self._merge_result(result)
        return str(self.state.get("bot_reply_text") or "").strip()

    async def submit_section_time_barge_in(
        self,
        *,
        partial_candidate_text: str = "",
    ) -> str:
        """Force the pending graph interrupt to leave an overrun section."""

        if self.state is None:
            await self.start_or_resume()
        if not self.state or self.state.get("should_close"):
            return ""

        graph = await get_graph()
        event = {
            "event_type": "section_time_barge_in",
            "text": " ".join(partial_candidate_text.split()),
            "elapsed_secs": self.elapsed_secs(),
            "received_at": utc_now_iso(),
        }
        result = await graph.ainvoke(
            Command(resume=event),
            config=self._config,
        )
        self._merge_result(result)
        return str(self.state.get("bot_reply_text") or "").strip()

    async def finalize_closing(self) -> None:
        """Complete durable lifecycle after closing audio has finished."""

        state = self.state or {}
        session_id = str(state.get("interview_session_id") or "")
        elapsed = self.elapsed_secs()
        await drain_background_persistence(session_id or None)
        if session_id:
            await interview_session_repository.complete_session(
                session_id,
                total_elapsed_secs=elapsed,
            )
        await assessment_context_repository.mark_candidate_completed(
            self.candidate_assessment_id
        )
        state.update(
            {
                "session_status": "COMPLETED",
                "closing_done": True,
                "elapsed_secs": elapsed,
                "remaining_secs": 0,
                "holistic_evaluation_status": (
                    state.get("holistic_evaluation_status") or "QUEUED"
                ),
            }
        )
        self.state = state
        logger.info(
            "Interview completed; holistic evaluation is queued",
            extra={
                "candidate_assessment_id": self.candidate_assessment_id,
                "interview_session_id": session_id,
            },
        )

    async def close(self, *, terminated: bool = False) -> str:
        """Finalize or terminate the active interview lifecycle."""

        if self.state is None:
            self.state = {}
        if terminated:
            self.state["session_status"] = "TERMINATED"
        else:
            await self.finalize_closing()
        return ""
