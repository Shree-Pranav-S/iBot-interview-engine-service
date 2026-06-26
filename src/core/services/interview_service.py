"""Interview orchestration service for LiveKit and LangGraph."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from src.core.services.graph_runner import InterviewGraphRunner
from src.data.repositories import interview_session_repository
from src.utils.interview_graph import deterministic_violation_id

TERMINAL_STATUSES = {"COMPLETED", "TERMINATED", "DEACTIVATED", "EVALUATED"}
GRACE_PERIOD_SECS = 300


class InterviewOrchestrator:
    """Compatibility adapter around InterviewGraphRunner."""

    def __init__(
        self,
        *,
        candidate_assessment_id: str,
        runner: InterviewGraphRunner | None = None,
    ) -> None:
        self.candidate_assessment_id = candidate_assessment_id
        self.runner = runner or InterviewGraphRunner()

    async def start_session(self) -> dict[str, Any]:
        return await self.runner.start(self.candidate_assessment_id)

    async def resume_session(self) -> dict[str, Any] | None:
        state = await self.runner.get_state(self.candidate_assessment_id)
        if not state:
            return None
        if (
            state.get("closing_done")
            or str(state.get("session_status") or "").upper() in TERMINAL_STATUSES
        ):
            return state

        session_id = state.get("interview_session_id")
        if session_id:
            await interview_session_repository.mark_session_in_progress(str(session_id))

        reconnect_count = int(state.get("reconnect_count") or 0) + 1
        replay_text = str(state.get("last_bot_text") or "").strip()
        return await self.runner.update_state(
            self.candidate_assessment_id,
            {
                "session_status": "IN_PROGRESS",
                "grace_period_expires_at": None,
                "disconnected_at": None,
                "reconnect_count": reconnect_count,
                "resumed": True,
                "bot_reply_text": replay_text,
                "bot_reply_type": "replay" if replay_text else "",
            },
        )

    async def submit_response(
        self,
        transcript: str,
        *,
        stt_confidence: float | None = None,
        duration_ms: int | None = None,
    ) -> dict[str, Any]:
        state = await self.runner.get_state(self.candidate_assessment_id)
        if state is None:
            await self.start_session()
        elif (
            state.get("closing_done")
            or str(state.get("session_status") or "").upper() in TERMINAL_STATUSES
        ):
            return state

        if transcript in {"__SILENCE__", "__TIME_UP__"}:
            event: dict[str, Any] | str = transcript
        else:
            event = {
                "event_type": "candidate_answer",
                "text": transcript,
                "stt_confidence": stt_confidence,
                "duration_ms": duration_ms,
                "ended_at": datetime.now(UTC).isoformat(),
                "eot_source": "livekit_turn_detector",
                "raw_metadata": {},
            }
        return await self.runner.resume(self.candidate_assessment_id, event)

    async def close_session(self, *, terminated: bool = False) -> dict[str, Any] | None:
        state = await self.get_current_state()
        if state is None or state.get("closing_done"):
            return state
        event_type = "candidate_disconnect" if terminated else "timer_expired"
        return await self.runner.resume(
            self.candidate_assessment_id,
            {"event_type": event_type, "text": ""},
        )

    async def pause_session(self) -> None:
        state = await self.get_current_state()
        if not state or state.get("closing_done"):
            return

        session_id = state.get("interview_session_id")
        if not session_id:
            return

        now = datetime.now(UTC)
        expires_at = now + timedelta(seconds=GRACE_PERIOD_SECS)
        await interview_session_repository.pause_session(str(session_id), expires_at)
        await self.runner.update_state(
            self.candidate_assessment_id,
            {
                "session_status": "PAUSED",
                "disconnected_at": now.isoformat(),
                "grace_period_expires_at": expires_at.isoformat(),
            },
        )

    async def expire_disconnect_grace(self) -> dict[str, Any] | None:
        state = await self.get_current_state()
        if not state or str(state.get("session_status") or "") != "PAUSED":
            return state

        expires_at_raw = state.get("grace_period_expires_at")
        if not expires_at_raw:
            return state
        expires_at = datetime.fromisoformat(str(expires_at_raw))
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if datetime.now(UTC) < expires_at:
            return state

        session_id = state.get("interview_session_id")
        if not session_id:
            return state
        turn_number = int(state.get("turn_number") or 0)
        await interview_session_repository.append_violation(
            str(session_id),
            {
                "violation_id": deterministic_violation_id(
                    str(session_id),
                    turn_number,
                    "disconnection_timeout",
                ),
                "turn_number": turn_number,
                "violation_type": "disconnection_timeout",
                "candidate_transcript": "",
                "severity": "high",
                "timestamp": datetime.now(UTC).isoformat(),
                "metadata": {"grace_period_expires_at": expires_at.isoformat()},
            },
        )
        await interview_session_repository.deactivate_session(
            str(session_id),
            reason="disconnect_grace_expired",
        )
        return await self.runner.update_state(
            self.candidate_assessment_id,
            {
                "session_status": "DEACTIVATED",
                "next_action": "deactivated",
                "should_close": True,
            },
        )

    async def get_current_state(self) -> dict[str, Any] | None:
        return await self.runner.get_state(self.candidate_assessment_id)
