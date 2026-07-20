"""Delegates candidate session lifecycle calls to core-api-service."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime

from src.clients.core_api_client import CoreApiClient, get_core_api_client
from src.schemas.livekit import (
    CandidateConnectionContext,
    CandidateSessionBootstrapResponse,
)


class CoreApiSessionService:
    """HTTP facade for core-api /internal/interview session routes."""

    def __init__(
        self,
        core_api_client_factory: Callable[[], CoreApiClient] = get_core_api_client,
    ) -> None:
        """Store the injectable core API client factory."""

        self._core_api_client_factory = core_api_client_factory

    def _client(self) -> CoreApiClient:
        return self._core_api_client_factory()

    async def enter_with_invitation(
        self,
        invitation_token: uuid.UUID,
    ) -> CandidateSessionBootstrapResponse:
        """Exchange an invitation for candidate session context."""

        return await self._client().enter_with_invitation(invitation_token)

    async def get_session_context(
        self,
        session_token: str,
    ) -> CandidateSessionBootstrapResponse:
        """Restore candidate context from a browser session token."""

        return await self._client().get_session_context(session_token)

    async def authorize_interview_connection(
        self,
        session_token: str,
    ) -> CandidateConnectionContext:
        """Authorize one candidate interview connection."""

        return await self._client().authorize_interview_connection(session_token)

    async def authorize_demo(
        self,
        session_token: str,
    ) -> dict[str, object]:
        """Authorize a disposable demo session."""

        return await self._client().authorize_demo(session_token)

    async def record_disconnect(
        self,
        *,
        session_id: uuid.UUID,
        connection_id: str,
        candidate_assessment_id: uuid.UUID,
        reason: str,
        elapsed_secs: int,
    ) -> dict[str, object]:
        """Persist a dropped candidate connection."""

        return await self._client().record_disconnect(
            session_id=session_id,
            connection_id=connection_id,
            candidate_assessment_id=candidate_assessment_id,
            reason=reason,
            elapsed_secs=elapsed_secs,
        )

    async def record_tab_switch(
        self,
        *,
        session_id: uuid.UUID,
        connection_id: str,
        candidate_assessment_id: uuid.UUID,
        event_id: uuid.UUID,
        occurred_at: datetime,
    ) -> dict[str, object]:
        """Persist one main-room tab switch through the internal core API."""

        return await self._client().record_tab_switch(
            session_id=session_id,
            connection_id=connection_id,
            candidate_assessment_id=candidate_assessment_id,
            event_id=event_id,
            occurred_at=occurred_at,
        )

    async def record_proctoring_event(
        self,
        *,
        session_id: uuid.UUID,
        connection_id: str,
        candidate_assessment_id: uuid.UUID,
        event_id: uuid.UUID,
        event_type: str,
        condition_started_at: datetime,
        observed_duration_ms: int,
        sample_count: int,
        max_face_count: int,
        min_confidence: float | None,
        max_confidence: float | None,
        source: str,
        detector_version: str,
        model_name: str,
    ) -> dict[str, object]:
        """Persist one face proctoring episode through the internal core API."""

        return await self._client().record_proctoring_event(
            session_id=session_id,
            connection_id=connection_id,
            candidate_assessment_id=candidate_assessment_id,
            event_id=event_id,
            event_type=event_type,
            condition_started_at=condition_started_at,
            observed_duration_ms=observed_duration_ms,
            sample_count=sample_count,
            max_face_count=max_face_count,
            min_confidence=min_confidence,
            max_confidence=max_confidence,
            source=source,
            detector_version=detector_version,
            model_name=model_name,
        )
