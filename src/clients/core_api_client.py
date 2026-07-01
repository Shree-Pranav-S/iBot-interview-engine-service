"""HTTP client for core-api internal interview persistence APIs."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, TypeVar, cast

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
T = TypeVar("T")
_INTERNAL_HEADER = "interview-engine"
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)
_HTTP_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)
_http_client: httpx.AsyncClient | None = None


def _create_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_TIMEOUT, limits=_HTTP_LIMITS)


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = _create_http_client()
    return _http_client


async def close_core_api_http_client() -> None:
    """Close the shared pooled HTTP client during process shutdown."""
    global _http_client
    if _http_client is not None:
        await _http_client.aclose()
        _http_client = None


def _exception_for_status(status_code: int, message: str) -> AppException:
    mapping: dict[int, type[AppException]] = {
        400: BadRequestException,
        401: AuthenticationException,
        403: ForbiddenException,
        404: NotFoundException,
        409: ConflictException,
        500: InternalServerException,
        502: BadGatewayException,
    }
    exc_type = mapping.get(status_code, AppException)
    return exc_type(message, status_code=status_code)


class CoreApiClient:
    """Async client for core-api /internal/interview routes."""

    def __init__(
        self,
        base_url: str | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base_url = (base_url or settings.CORE_API_URL).rstrip("/")
        self._http_client = http_client

    def _client(self) -> httpx.AsyncClient:
        return self._http_client or _get_http_client()

    def _headers(self) -> dict[str, str]:
        return {"X-Internal-Service": _INTERNAL_HEADER}

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
                headers=self._headers(),
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
        data = await self._request(
            "POST",
            "/internal/interview/session/authorize-connection",
            json={"session_token": session_token},
        )
        return CandidateConnectionContext.model_validate(data)

    async def authorize_demo(self, session_token: str) -> dict[str, Any]:
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
        await self._request(
            "POST",
            "/internal/interview/events",
            json=event.model_dump(mode="json"),
        )

    async def initialize_interview_context(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
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
        await self._request(
            "POST",
            f"/internal/interview/sessions/{session_id}/complete",
            json={"total_elapsed_secs": total_elapsed_secs},
        )

    async def mark_candidate_timer_started(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> datetime:
        data = await self._request(
            "POST",
            f"/internal/interview/candidates/{candidate_assessment_id}/timer-started",
        )
        return datetime.fromisoformat(str(data["started_at"]))

    async def mark_candidate_completed(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> None:
        await self._request(
            "POST",
            f"/internal/interview/candidates/{candidate_assessment_id}/completed",
        )

    async def load_evaluation_source(
        self,
        candidate_assessment_id: str | uuid.UUID,
    ) -> dict[str, Any] | None:
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
    global _default_client
    if _default_client is None:
        _default_client = CoreApiClient()
    return _default_client
