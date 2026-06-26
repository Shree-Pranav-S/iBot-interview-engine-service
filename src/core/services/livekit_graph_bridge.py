"""Bridge between LiveKit voice turns and the LangGraph interview workflow."""

from __future__ import annotations

from typing import Any

from src.core.services.interview_service import InterviewOrchestrator
from src.data.repositories import assessment_context_repository
from src.utils.interview_graph import utc_now_iso
from src.utils.interview_turns import extract_bot_reply


class LiveKitInterviewBridge:
    """Small stateful adapter around the existing interview orchestrator."""

    def __init__(self, *, candidate_assessment_id: str) -> None:
        self.candidate_assessment_id = candidate_assessment_id
        self.orchestrator = InterviewOrchestrator(
            candidate_assessment_id=candidate_assessment_id
        )
        self.state: dict[str, Any] | None = None

    async def start_or_resume(self) -> str:
        """Start or resume the LangGraph interview and return new bot speech."""

        previous = self.state
        state = await self.orchestrator.resume_session()
        if state is None:
            state = await self.orchestrator.start_session()

        self.state = state
        return extract_bot_reply(previous, state)

    async def start_timer(self) -> bool:
        """Start the durable time budget when the interviewer first speaks."""

        state = self.state or await self.orchestrator.get_current_state()
        if not state or state.get("started_at"):
            self.state = state
            return False

        started_at = utc_now_iso()
        self.state = await self.orchestrator.runner.update_state(
            self.candidate_assessment_id,
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
        state = await self.orchestrator.submit_response(
            text,
            stt_confidence=stt_confidence,
            duration_ms=duration_ms,
        )

        self.state = state
        return extract_bot_reply(previous, state)

    async def submit_silence(self) -> str:
        """Submit a graph silence sentinel after LiveKit user-away timeout."""

        return await self.submit_candidate_turn("__SILENCE__")

    async def close(self, *, terminated: bool = False) -> str:
        """Close the graph session and return any closing speech."""

        previous = self.state
        state = await self.orchestrator.close_session(terminated=terminated)
        if state is None:
            return ""

        self.state = state
        return extract_bot_reply(previous, state)
