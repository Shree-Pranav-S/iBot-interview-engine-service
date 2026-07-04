"""HTTP client for core-api internal interview persistence APIs."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, cast

import httpx

from src.config.settings import settings
from src.core.exceptions import (
    AppException,
    AuthenticationException,
    BadGatewayException,
    BadRequestException,
    ConflictException,
    ForbiddenException,
    InternalServerException,
    NotFoundException,
)
from src.schemas.evaluation_llm import FinalEvaluationRecord
from src.schemas.event_log import EventLogCreate
from src.schemas.livekit import (
    CandidateConnectionContext,
    CandidateSessionBootstrapResponse,
)

logger = logging.getLogger(__name__)
_INTERNAL_HEADERS = {"X-Internal-Service": "interview-engine"}
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
_HTTP_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)
_STATUS_EXCEPTIONS: dict[int, type[AppException]] = {
    400: BadRequestException,
    401: AuthenticationException,
    403: ForbiddenException,
    404: NotFoundException,
    409: ConflictException,
    500: InternalServerException,
    502: BadGatewayException,
}
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=_TIMEOUT, limits=_HTTP_LIMITS)
    return _http_client


async def close_core_api_http_client() -> None:
    """Close the shared pooled HTTP client during process shutdown."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def _exception_for_status(status_code: int, message: str) -> AppException:
    exc_type = _STATUS_EXCEPTIONS.get(status_code, AppException)
    return exc_type(message, status_code=status_code)


class CoreApiClient:
    """Async client for core-api /internal/interview routes."""

    def __init__(
        self,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Configure an optional base URL and injectable HTTP transport."""

        self._base_url = (base_url or settings.CORE_API_URL).rstrip("/")
        self._http_client = http_client

    def _client(self) -> httpx.AsyncClient:
        return self._http_client or _get_http_client()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        try:
            response = await self._client().request(
                method,
                url,
                headers=_INTERNAL_HEADERS,
                json=json,
                params=params,
            )
        except httpx.RequestError as exc:
            logger.exception("core-api request failed", extra={"url": url})
            raise BadGatewayException(
                f"core-api unreachable: {exc}",
            ) from exc

        if response.status_code >= 400:
            message = "core-api request failed"
            try:
                payload = response.json()
                if isinstance(payload, dict) and payload.get("message"):
                    message = str(payload["message"])
            except ValueError:
                message = response.text or message
            raise _exception_for_status(response.status_code, message)

        payload = response.json()
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"]
        return payload

    async def enter_with_invitation(
        self,
        invitation_token: uuid.UUID,
    ) -> CandidateSessionBootstrapResponse:
        """Exchange a one-time invitation for candidate session context."""

        data = await self._request(
            "POST",
            "/internal/interview/session/enter",
            json={"invitation_token": str(invitation_token)},
        )
        return CandidateSessionBootstrapResponse.model_validate(data)

    async def get_session_context(
        self,
        session_token: str,
    ) -> CandidateSessionBootstrapResponse:
        """Restore candidate session context from a session token."""

        data = await self._request(
            "POST",
            "/internal/interview/session/context",
            json={"session_token": session_token},
        )
        return CandidateSessionBootstrapResponse.model_validate(data)

    async def authorize_interview_connection(
        self,
        session_token: str,
    ) -> CandidateConnectionContext:
        """Authorize one LiveKit interview connection."""

        data = await self._request(
            "POST",
            "/internal/interview/session/authorize-connection",
            json={"session_token": session_token},
        )
        return CandidateConnectionContext.model_validate(data)

    async def authorize_demo(self, session_token: str) -> dict[str, Any]:
        """Authorize access to a disposable demo room."""

        return cast(
            dict[str, Any],
            await self._request(
                "POST",
                "/internal/interview/session/authorize-demo",
                json={"session_token": session_token},
            ),
        )

    async def record_disconnect(
        self,
        *,
        session_id: uuid.UUID,
        connection_id: str,
        candidate_assessment_id: uuid.UUID,
        reason: str,
        elapsed_secs: int,
    ) -> dict[str, Any]:
        """Persist a candidate connection drop."""

        return cast(
            dict[str, Any],
            await self._request(
                "POST",
                "/internal/interview/session/disconnect",
                json={
                    "session_id": str(session_id),
                    "connection_id": connection_id,
                    "candidate_assessment_id": str(candidate_assessment_id),
                    "reason": reason,
                    "elapsed_secs": elapsed_secs,
                },
            ),
        )

    async def create_event_log(self, event: EventLogCreate) -> None:
        """Persist one durable interview-engine event."""

        await self._request(
            "POST",
            "/internal/interview/events",
            json=event.model_dump(mode="json"),
        )

    async def initialize_interview_context(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load or create the graph's initial context and session."""

        data = await self._request(
            "POST",
            "/internal/interview/context/initialize",
            json={"candidate_assessment_id": str(candidate_assessment_id)},
        )
        return data["context"], data["session"]

    async def load_interview_context(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> dict[str, Any]:
        """Load current interview context for a candidate assessment."""

        return cast(
            dict[str, Any],
            await self._request(
                "GET",
                f"/internal/interview/context/{candidate_assessment_id}",
            ),
        )

    async def get_session_by_candidate_assessment_id(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> dict[str, Any]:
        """Load the session associated with a candidate assessment."""

        return cast(
            dict[str, Any],
            await self._request(
                "GET",
                f"/internal/interview/sessions/by-ca/{candidate_assessment_id}",
            ),
        )

    async def persist_turn(
        self,
        *,
        session_id: str,
        transcript_items: list[dict[str, Any]],
        violations: list[dict[str, Any]],
        elapsed_secs: int | None = None,
    ) -> None:
        """Persist transcript items, violations, and optional elapsed time."""

        body: dict[str, Any] = {
            "transcript_items": transcript_items,
            "violations": violations,
        }
        if elapsed_secs is not None:
            body["elapsed_secs"] = elapsed_secs
        await self._request(
            "POST",
            f"/internal/interview/sessions/{session_id}/persist-turn",
            json=body,
        )

    async def mark_session_in_progress(self, session_id: str) -> None:
        """Mark a session as actively interviewing."""

        await self._request(
            "POST",
            f"/internal/interview/sessions/{session_id}/in-progress",
        )

    async def complete_session(
        self,
        session_id: str,
        *,
        total_elapsed_secs: int,
    ) -> None:
        """Complete a session with its authoritative elapsed time."""

        await self._request(
            "POST",
            f"/internal/interview/sessions/{session_id}/complete",
            json={"total_elapsed_secs": total_elapsed_secs},
        )

    async def mark_candidate_timer_started(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> datetime:
        """Persist and return the candidate's timer start time."""

        data = await self._request(
            "POST",
            f"/internal/interview/candidates/{candidate_assessment_id}/timer-started",
        )
        return datetime.fromisoformat(str(data["started_at"]))

    async def mark_candidate_completed(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> None:
        """Mark the candidate assessment as completed."""

        await self._request(
            "POST",
            f"/internal/interview/candidates/{candidate_assessment_id}/completed",
        )

    async def load_evaluation_source(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> dict[str, Any] | None:
        """Load all source data required for holistic evaluation."""

        return cast(
            dict[str, Any] | None,
            await self._request(
                "GET",
                f"/internal/interview/evaluation/source/{candidate_assessment_id}",
            ),
        )

    async def evaluation_exists_for_hash(
        self,
        candidate_assessment_id: str | uuid.UUID,
        transcript_hash: str,
    ) -> bool:
        """Return whether this transcript fingerprint was already evaluated."""

        return bool(
            await self._request(
                "GET",
                "/internal/interview/evaluation/exists",
                params={
                    "candidate_assessment_id": str(candidate_assessment_id),
                    "transcript_hash": transcript_hash,
                },
            )
        )

    async def mark_evaluation_failed(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> None:
        """Persist a terminal holistic-evaluation failure."""

        await self._request(
            "POST",
            f"/internal/interview/evaluation/failed/{candidate_assessment_id}",
        )

    async def save_final_evaluation(
        self,
        record: FinalEvaluationRecord,
        *,
        recruiter_email: str,
    ) -> dict[str, Any]:
        """Persist a final evaluation and create its recruiter notification."""

        return cast(
            dict[str, Any],
            await self._request(
                "POST",
                "/internal/interview/evaluation/final",
                json={
                    "record": record.model_dump(mode="json"),
                    "recruiter_email": recruiter_email,
                },
            ),
        )

    async def health_ping(self) -> bool:
        """Return whether the core API health endpoint responds successfully."""

        try:
            response = await self._client().get(
                f"{self._base_url}/health",
                timeout=httpx.Timeout(5.0),
            )
            return response.status_code == 200
        except httpx.RequestError:
            return False


_default_client: CoreApiClient | None = None


def get_core_api_client() -> CoreApiClient:
    """Return the process-wide core API client wrapper."""

    global _default_client
    if _default_client is None:
        _default_client = CoreApiClient()
    return _default_client
