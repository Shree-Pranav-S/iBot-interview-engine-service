"""Two-token candidate entry and bounded LiveKit reconnection lifecycle."""

from __future__ import annotations

import uuid
from collections.abc import Callable

from src.clients.core_api_client import CoreApiClient, get_core_api_client
from src.schemas.livekit import (
    CandidateConnectionContext,
    CandidateSessionBootstrapResponse,
)


class CandidateSessionService:
    """Delegates session lifecycle persistence to core-api-service."""

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
