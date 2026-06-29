"""Two-token candidate entry and bounded LiveKit reconnection lifecycle."""

from __future__ import annotations

import logging
import secrets
import uuid
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

from src.core.exceptions import AuthenticationException, ForbiddenException
from src.data.repositories.event_logs_repository import EventLogsRepository
from src.data.repositories.unit_of_work import InterviewUnitOfWork
from src.schemas.event_log import EventLogCreate, EventName, EventSource
from src.schemas.livekit import (
    CandidateConnectionContext,
    CandidateSessionBootstrapResponse,
)

logger = logging.getLogger(__name__)
RECONNECT_WINDOW = timedelta(minutes=5)
TERMINAL_SESSION_STATUSES = {
    "COMPLETED",
    "EVALUATED",
    "EVALUATION_FAILED",
    "DEACTIVATED",
    "TERMINATED",
}
TERMINAL_CANDIDATE_STATUSES = {"COMPLETED", "EVALUATED", "TERMINATED"}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _aware(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=UTC)


def _sections_overview(context: dict[str, Any]) -> list[str]:
    plan = context.get("interview_plan")
    if not isinstance(plan, dict) or not isinstance(plan.get("sections"), list):
        return []
    return [
        str(section["section_name"])
        for section in plan["sections"]
        if isinstance(section, dict) and section.get("section_name")
    ]


class CandidateSessionService:
    """Own invitation consumption, session authentication, and reconnect rules."""

    def __init__(
        self,
        unit_of_work_factory: Callable[[], InterviewUnitOfWork] = InterviewUnitOfWork,
    ) -> None:
        self._unit_of_work_factory = unit_of_work_factory

    @staticmethod
    async def _record(
        repository: EventLogsRepository,
        *,
        event_name: EventName,
        correlation_id: str,
        candidate_assessment_id: uuid.UUID | None,
        metadata: dict[str, Any],
    ) -> None:
        await repository.create(
            EventLogCreate(
                event_name=event_name,
                source_service=EventSource.INTERVIEW_ENGINE,
                correlation_id=correlation_id,
                candidate_assessment_id=candidate_assessment_id,
                metadata=metadata,
            )
        )

    @staticmethod
    def _bootstrap_response(
        context: dict[str, Any],
        session: dict[str, Any],
        *,
        invite_reissued: bool,
    ) -> CandidateSessionBootstrapResponse:
        expires_at = _aware(session.get("session_token_expires_at"))
        if expires_at is None:
            raise RuntimeError("Session token expiry is missing")
        return CandidateSessionBootstrapResponse(
            session_token=str(session["session_token"]),
            session_token_expires_at=expires_at,
            session_status=str(session["status"]),
            invite_reissued=invite_reissued,
            interview_started=context.get("interview_started_at") is not None,
            reconnect_deadline=_aware(session.get("reconnect_deadline")),
            disconnect_count=int(session.get("disconnect_count") or 0),
            candidate_name=str(context.get("candidate_name") or "Candidate"),
            company_name=str(context.get("company_name") or "the company"),
            assessment_title=str(context.get("assessment_title") or "Interview"),
            interview_duration_mins=int(context.get("interview_duration_mins") or 0),
            window_end=_aware(context.get("window_end")) or _utc_now(),
            status=str(context.get("candidate_assessment_status") or ""),
            sections_overview=_sections_overview(context),
        )

    async def enter_with_invitation(
        self,
        invitation_token: uuid.UUID,
    ) -> CandidateSessionBootstrapResponse:
        """Consume an invitation once or return its existing active session token."""

        rejection: Exception | None = None
        response: CandidateSessionBootstrapResponse | None = None
        async with self._unit_of_work_factory() as unit_of_work:
            repository = unit_of_work.candidate_sessions
            event_repository = unit_of_work.event_logs
            context = await repository.lock_invitation_context(invitation_token)

            if not context:
                await self._record(
                    event_repository,
                    event_name=EventName.INVITE_LINK_REJECTED,
                    correlation_id=str(invitation_token),
                    candidate_assessment_id=None,
                    metadata={"reason": "invitation_not_found"},
                )
                rejection = AuthenticationException("Invalid interview invitation.")
            else:
                candidate_assessment_id = context["candidate_assessment_id"]
                session = await repository.get_or_create_session_for_update(
                    candidate_assessment_id
                )
                session_status = str(session.get("status") or "").upper()
                candidate_status = str(
                    context.get("candidate_assessment_status") or ""
                ).upper()
                correlation_id = str(session["id"])

                if (
                    session_status in TERMINAL_SESSION_STATUSES
                    or candidate_status in TERMINAL_CANDIDATE_STATUSES
                ):
                    await self._record(
                        event_repository,
                        event_name=EventName.INVITE_LINK_REJECTED,
                        correlation_id=correlation_id,
                        candidate_assessment_id=candidate_assessment_id,
                        metadata={
                            "reason": "session_terminal",
                            "session_status": session_status,
                            "candidate_assessment_status": candidate_status,
                        },
                    )
                    rejection = ForbiddenException(
                        "This interview session is already closed."
                    )
                else:
                    now = _utc_now()
                    expires_at = _aware(session.get("session_token_expires_at"))
                    invite_consumed = bool(context.get("invite_consumed"))
                    existing_token = str(session.get("session_token") or "")

                    if invite_consumed and expires_at is not None and expires_at <= now:
                        await self._record(
                            event_repository,
                            event_name=EventName.INVITE_LINK_REJECTED,
                            correlation_id=correlation_id,
                            candidate_assessment_id=candidate_assessment_id,
                            metadata={
                                "reason": "session_token_expired",
                                "session_status": session_status,
                            },
                        )
                        rejection = AuthenticationException(
                            "This interview session has expired."
                        )
                    else:
                        invite_reissued = invite_consumed
                        if not existing_token:
                            duration_mins = max(
                                1,
                                int(context.get("interview_duration_mins") or 0),
                            )
                            minimum_expiry = now + timedelta(minutes=duration_mins + 60)
                            window_end = _aware(context.get("window_end"))
                            expires_at = max(
                                minimum_expiry,
                                (window_end + timedelta(hours=1))
                                if window_end is not None
                                else minimum_expiry,
                            )
                            existing_token = secrets.token_urlsafe(48)
                            await repository.set_session_credentials(
                                session["id"],
                                session_token=existing_token,
                                expires_at=expires_at,
                            )
                            session.update(
                                {
                                    "session_token": existing_token,
                                    "session_token_expires_at": expires_at,
                                }
                            )

                        if not invite_consumed:
                            await repository.consume_invitation(candidate_assessment_id)
                            event_name = EventName.INVITE_LINK_CONSUMED
                        else:
                            event_name = EventName.SESSION_TOKEN_REISSUED

                        await self._record(
                            event_repository,
                            event_name=event_name,
                            correlation_id=correlation_id,
                            candidate_assessment_id=candidate_assessment_id,
                            metadata={
                                "session_status": session_status,
                                "session_token_expires_at": (
                                    expires_at.isoformat()
                                    if expires_at is not None
                                    else None
                                ),
                            },
                        )
                        response = self._bootstrap_response(
                            context,
                            session,
                            invite_reissued=invite_reissued,
                        )

        if rejection is not None:
            raise rejection
        if response is None:
            raise RuntimeError("Candidate session entry produced no result")
        return response

    async def get_session_context(
        self,
        session_token: str,
    ) -> CandidateSessionBootstrapResponse:
        """Load waiting-room context using only the browser session credential."""

        rejection: Exception | None = None
        response: CandidateSessionBootstrapResponse | None = None
        async with self._unit_of_work_factory() as unit_of_work:
            repository = unit_of_work.candidate_sessions
            context = await repository.lock_session_context_by_token(session_token)
            rejection = self._validate_active_token(context)
            if rejection is None:
                session = {
                    **context,
                    "id": context["session_id"],
                    "status": context["session_status"],
                }
                response = self._bootstrap_response(
                    context,
                    session,
                    invite_reissued=False,
                )

        if rejection is not None:
            raise rejection
        if response is None:
            raise RuntimeError("Candidate session context produced no result")
        return response

    @staticmethod
    def _validate_active_token(
        context: dict[str, Any],
    ) -> Exception | None:
        if not context:
            return AuthenticationException("Invalid interview session token.")
        expires_at = _aware(context.get("session_token_expires_at"))
        if expires_at is None or expires_at <= _utc_now():
            return AuthenticationException("This interview session has expired.")
        if (
            str(context.get("session_status") or "").upper()
            in TERMINAL_SESSION_STATUSES
            or str(context.get("candidate_assessment_status") or "").upper()
            in TERMINAL_CANDIDATE_STATUSES
        ):
            return ForbiddenException("This interview session is already closed.")
        return None

    async def authorize_interview_connection(
        self,
        session_token: str,
    ) -> CandidateConnectionContext:
        """Validate a session token, lazily enforce timeout, and bind a connection."""

        rejection: Exception | None = None
        response: CandidateConnectionContext | None = None
        async with self._unit_of_work_factory() as unit_of_work:
            repository = unit_of_work.candidate_sessions
            event_repository = unit_of_work.event_logs
            context = await repository.lock_session_context_by_token(session_token)
            rejection = self._validate_active_token(context)

            if rejection is None:
                now = _utc_now()
                session_id = context["session_id"]
                candidate_assessment_id = context["candidate_assessment_id"]
                session_status = str(context.get("session_status") or "").upper()

                if session_status == "DISCONNECTED":
                    reconnect_deadline = _aware(context.get("reconnect_deadline"))
                    if reconnect_deadline is None or reconnect_deadline < now:
                        terminated = await repository.terminate_reconnect_timeout(
                            session_id
                        )
                        timeout_count = int(
                            terminated.get("timeout_disconnect_count") or 0
                        )
                        await self._record(
                            event_repository,
                            event_name=(EventName.SESSION_DISCONNECT_TIMEOUT_RECORDED),
                            correlation_id=str(session_id),
                            candidate_assessment_id=candidate_assessment_id,
                            metadata={
                                "timeout_disconnect_count": timeout_count,
                                "reconnect_deadline": (
                                    reconnect_deadline.isoformat()
                                    if reconnect_deadline
                                    else None
                                ),
                            },
                        )
                        await self._record(
                            event_repository,
                            event_name=(EventName.SESSION_TERMINATED_RECONNECT_TIMEOUT),
                            correlation_id=str(session_id),
                            candidate_assessment_id=candidate_assessment_id,
                            metadata={
                                "timeout_disconnect_count": timeout_count,
                                "disconnect_count": int(
                                    terminated.get("disconnect_count") or 0
                                ),
                            },
                        )
                        rejection = ForbiddenException(
                            "The five-minute reconnection window has expired."
                        )
                    else:
                        restored = await repository.restore_disconnected_session(
                            session_id
                        )
                        await self._record(
                            event_repository,
                            event_name=EventName.SESSION_RECONNECTED,
                            correlation_id=str(session_id),
                            candidate_assessment_id=candidate_assessment_id,
                            metadata={
                                "disconnect_count": int(
                                    restored.get("disconnect_count") or 0
                                ),
                                "timeout_disconnect_count": int(
                                    restored.get("timeout_disconnect_count") or 0
                                ),
                                "reconnected_at": now.isoformat(),
                            },
                        )

                if rejection is None:
                    # A new token request for an in-progress interview replaces
                    # the browser/agent connection even if the old LiveKit close
                    # callback has not reached us yet. Rotating the generation
                    # makes that eventual callback stale and lets the token
                    # service explicitly dispatch a fresh agent immediately.
                    active_connection_id = str(
                        context.get("active_connection_id") or ""
                    )
                    connection_id = (
                        uuid.uuid4().hex
                        if session_status == "IN_PROGRESS"
                        else active_connection_id or uuid.uuid4().hex
                    )
                    await repository.bind_active_connection(
                        context["session_id"],
                        connection_id=connection_id,
                    )
                    response = CandidateConnectionContext(
                        session_id=context["session_id"],
                        connection_id=connection_id,
                        candidate_assessment_id=context["candidate_assessment_id"],
                        candidate_id=context["candidate_id"],
                        assessment_id=context["assessment_id"],
                        candidate_name=str(
                            context.get("candidate_name") or "Candidate"
                        ),
                        session_token_expires_at=_aware(
                            context["session_token_expires_at"]
                        )
                        or now,
                        elapsed_secs=max(
                            0,
                            int(context.get("total_elapsed_secs") or 0),
                        ),
                        interview_started=(
                            context.get("interview_started_at") is not None
                        ),
                    )

        if rejection is not None:
            raise rejection
        if response is None:
            raise RuntimeError("Interview connection authorization failed")
        return response

    async def authorize_demo(
        self,
        session_token: str,
    ) -> dict[str, Any]:
        """Validate demo access without changing real interview lifecycle state."""

        rejection: Exception | None = None
        context: dict[str, Any] = {}
        async with self._unit_of_work_factory() as unit_of_work:
            repository = unit_of_work.candidate_sessions
            context = await repository.lock_session_context_by_token(session_token)
            rejection = self._validate_active_token(context)
            if (
                rejection is None
                and str(context.get("session_status") or "").upper() == "DISCONNECTED"
            ):
                rejection = ForbiddenException(
                    "Resume the active interview instead of starting a demo."
                )

        if rejection is not None:
            raise rejection
        return context

    async def record_disconnect(
        self,
        *,
        session_id: uuid.UUID,
        connection_id: str,
        candidate_assessment_id: uuid.UUID,
        reason: str,
        elapsed_secs: int,
    ) -> dict[str, Any]:
        """Record one current LiveKit drop, ignoring stale replaced jobs."""

        reconnect_deadline = _utc_now() + RECONNECT_WINDOW
        outcome: dict[str, Any] = {}
        async with self._unit_of_work_factory() as unit_of_work:
            repository = unit_of_work.candidate_sessions
            event_repository = unit_of_work.event_logs
            outcome = await repository.record_disconnect(
                session_id,
                connection_id=connection_id,
                reconnect_deadline=reconnect_deadline,
                elapsed_secs=elapsed_secs,
            )
            if not outcome:
                logger.info(
                    "Ignored stale or terminal LiveKit disconnect",
                    extra={
                        "session_id": str(session_id),
                        "connection_id": connection_id,
                    },
                )
                return {}

            disconnect_count = int(outcome.get("disconnect_count") or 0)
            if str(outcome.get("status")) == "TERMINATED":
                event_name = EventName.SESSION_TERMINATED_MAX_DISCONNECTS
                metadata = {
                    "reason": reason or "livekit_session_closed",
                    "disconnect_count": disconnect_count,
                    "elapsed_secs": max(0, elapsed_secs),
                }
            else:
                event_name = EventName.SESSION_DISCONNECTED
                metadata = {
                    "reason": reason or "livekit_session_closed",
                    "disconnect_count": disconnect_count,
                    "timeout_disconnect_count": int(
                        outcome.get("timeout_disconnect_count") or 0
                    ),
                    "reconnect_deadline": reconnect_deadline.isoformat(),
                    "elapsed_secs": max(0, elapsed_secs),
                }
            await self._record(
                event_repository,
                event_name=event_name,
                correlation_id=str(session_id),
                candidate_assessment_id=candidate_assessment_id,
                metadata=metadata,
            )
        return outcome
