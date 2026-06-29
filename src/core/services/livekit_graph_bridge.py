"""Stateful adapter between LiveKit voice events and the interview LangGraph."""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
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
from src.core.services.candidate_session_service import CandidateSessionService
from src.core.services.event_log_service import (
    try_record_event_in_background,
)
from src.data.repositories.unit_of_work import InterviewUnitOfWork
from src.schemas.event_log import EventLogCreate, EventName, EventSource
from src.utils.interview_graph import utc_now_iso

logger = logging.getLogger(__name__)
TERMINAL_STATUSES = {"COMPLETED", "TERMINATED", "DEACTIVATED", "EVALUATED"}


class LiveKitInterviewBridge:
    """Resume one checkpointed graph thread from LiveKit-confirmed events."""

    def __init__(
        self,
        *,
        candidate_assessment_id: str,
        interview_session_id: str,
        connection_id: str,
    ) -> None:
        self.candidate_assessment_id = candidate_assessment_id
        self.interview_session_id = interview_session_id
        self.connection_id = connection_id
        self.state: dict[str, Any] | None = None
        self._config = {"configurable": {"thread_id": self.candidate_assessment_id}}
        self._timer_started_monotonic: float | None = None
        self._timer_started_at: datetime | None = None
        self._elapsed_before_connection_secs = 0

    @staticmethod
    def _is_terminal(state: dict[str, Any]) -> bool:
        """
        Check if the interview state represents a terminal session status.

        Args:
            state: The current interview state dictionary.

        Returns:
            True if the session is completed, terminated, deactivated, or evaluated.
        """
        return str(state.get("session_status") or "").upper() in TERMINAL_STATUSES

    async def _checkpoint_state(self) -> dict[str, Any]:
        """
        Retrieve the latest state snapshot from the LangGraph checkpointer.

        Returns:
            The raw dictionary state from the latest checkpoint, or an empty dict.
        """
        graph = await get_graph()
        snapshot = await graph.aget_state(self._config)
        values = getattr(snapshot, "values", None)
        return dict(values or {})

    async def start_or_resume(self) -> str:
        """
        Initialize context and opening, or restore the current spoken prompt.
        This handles both brand new interviews and recovering from disconnects.

        Returns:
            The text the bot should say to resume or start the conversation.
        """

        graph = await get_graph()
        existing = await self._checkpoint_state()
        if existing:
            self.state = existing
            async with InterviewUnitOfWork() as unit_of_work:
                context = await unit_of_work.assessment_context.load_interview_context(
                    self.candidate_assessment_id,
                )
                session = await unit_of_work.interview_sessions.get_session_by_candidate_assessment_id(
                    self.candidate_assessment_id,
                )
            self._elapsed_before_connection_secs = max(
                0,
                int(session.get("total_elapsed_secs") or 0),
            )
            started_at = context.get("interview_started_at")
            if isinstance(started_at, datetime):
                self._timer_started_at = started_at
                self._timer_started_monotonic = time.monotonic()
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
        """
        Persist the timer only when LiveKit says first bot playout has begun.
        This ensures the candidate's timer doesn't run while they are loading.

        Returns:
            True if the timer was successfully started now, False if already started.
        """

        if self.state and self.state.get("timer_started"):
            return False

        session_id = str((self.state or {}).get("interview_session_id") or "")
        async with InterviewUnitOfWork() as unit_of_work:
            started_at = (
                await unit_of_work.assessment_context.mark_candidate_timer_started(
                    self.candidate_assessment_id,
                )
            )
            if session_id:
                await unit_of_work.interview_sessions.mark_session_in_progress(
                    session_id,
                )
        self._timer_started_monotonic = time.monotonic()
        self._timer_started_at = started_at
        if session_id:
            await try_record_event_in_background(
                EventLogCreate(
                    event_name=EventName.INTERVIEW_STARTED,
                    source_service=EventSource.INTERVIEW_ENGINE,
                    correlation_id=session_id,
                    candidate_assessment_id=uuid.UUID(self.candidate_assessment_id),
                    metadata={
                        "timer_started_at": started_at.isoformat(),
                    },
                )
            )

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
        """
        Return official elapsed interview time without starting the clock.
        Accounts for time across multiple reconnects.

        Returns:
            The total elapsed time in seconds.
        """

        if self._timer_started_monotonic is not None:
            return max(
                0,
                self._elapsed_before_connection_secs
                + int(time.monotonic() - self._timer_started_monotonic),
            )
        return max(0, int((self.state or {}).get("elapsed_secs") or 0))

    def seconds_until_section_barge_in(self) -> float | None:
        """
        Return delay to the grace deadline without stealing later sections.
        This is used to determine when the agent should forcefully move on.

        Returns:
            The number of seconds until a barge-in should occur, or None if inapplicable.
        """

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
        """
        Merge the new LangGraph state result with local timer properties.

        Args:
            result: The new state dictionary returned by LangGraph.
        """
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
        """
        Resume the pending graph interrupt with one final speech transcript.
        This triggers a state machine transition to evaluate the response and
        generate the next question.

        Args:
            text: The candidate's spoken response.
            stt_confidence: The confidence score from the STT provider.
            duration_ms: How long the candidate spoke.

        Returns:
            The text for the bot to speak in reply.
        """

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
        """
        Resume with LiveKit's five-second no-speech event; no LLM is called.
        This allows the graph to naturally handle candidate silence.

        Returns:
            The text for the bot to speak to prompt the candidate.
        """

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
        """
        Force the pending graph interrupt to leave an overrun section.
        This is triggered when the candidate takes too long on a section.

        Args:
            partial_candidate_text: Whatever text the candidate had spoken so far.

        Returns:
            The text for the bot to speak (usually a transition to the next section).
        """

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
        """
        Complete durable lifecycle after closing audio has finished.
        This marks the session completed and triggers downstream tasks like evaluation.
        """

        state = self.state or {}
        session_id = str(state.get("interview_session_id") or "")
        elapsed = self.elapsed_secs()
        await drain_background_persistence(session_id or None)
        async with InterviewUnitOfWork() as unit_of_work:
            if session_id:
                await unit_of_work.interview_sessions.complete_session(
                    session_id,
                    total_elapsed_secs=elapsed,
                )
            await unit_of_work.assessment_context.mark_candidate_completed(
                self.candidate_assessment_id,
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
        await try_record_event_in_background(
            EventLogCreate(
                event_name=EventName.INTERVIEW_ENDED,
                source_service=EventSource.INTERVIEW_ENGINE,
                correlation_id=session_id or self.candidate_assessment_id,
                candidate_assessment_id=uuid.UUID(self.candidate_assessment_id),
                metadata={
                    "end_reason": "completed",
                    "session_status": "COMPLETED",
                    "elapsed_secs": elapsed,
                },
                duration_ms=elapsed * 1000,
            )
        )
        logger.info(
            "Interview completed; holistic evaluation is queued",
            extra={
                "candidate_assessment_id": self.candidate_assessment_id,
                "interview_session_id": session_id,
            },
        )

    async def record_disconnect(self, reason: str) -> None:
        """
        Record an unexpected LiveKit drop under the bounded reconnect policy.

        Args:
            reason: The disconnect reason code.
        """

        state = self.state or {}
        if (
            self._is_terminal(state)
            or state.get("should_close")
            or state.get("closing_done")
        ):
            return

        if not self.interview_session_id or not self.connection_id:
            logger.warning(
                "Cannot record disconnect without session connection metadata",
                extra={
                    "candidate_assessment_id": self.candidate_assessment_id,
                },
            )
            return
        await CandidateSessionService().record_disconnect(
            session_id=uuid.UUID(self.interview_session_id),
            connection_id=self.connection_id,
            candidate_assessment_id=uuid.UUID(self.candidate_assessment_id),
            reason=reason,
            elapsed_secs=self.elapsed_secs(),
        )
