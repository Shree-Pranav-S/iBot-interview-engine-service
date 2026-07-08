"""Business logic for issuing LiveKit interview room tokens."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

from livekit import api

from src.config.settings import settings
from src.core.exceptions import (
    LiveKitConfigurationException,
    SessionTokenExpiryInvalidException,
    SessionTokenExpiryMissingException,
)
from src.core.services.core_api_session_service import CoreApiSessionService
from src.schemas.livekit import (
    CandidateConnectionContext,
    LiveKitTokenRequest,
    LiveKitTokenResponse,
)
from src.utils.livekit import (
    candidate_participant_identity,
    candidate_room_name,
    demo_room_name,
)


class LiveKitTokenService:
    """Service layer for candidate LiveKit room credentials."""

    def __init__(
        self,
        core_api_session_service: CoreApiSessionService | None = None,
    ) -> None:
        """Configure the core-api session authorization dependency."""

        self._core_api_session_service = (
            core_api_session_service or CoreApiSessionService()
        )

    @staticmethod
    def _validate_configuration() -> None:
        """
        Ensure all required LiveKit configuration variables are present.

        Raises:
            LiveKitConfigurationException: If the URL, API key, or API secret are missing.
        """
        if not settings.LIVEKIT_URL:
            raise LiveKitConfigurationException("LiveKit URL is not configured.")
        if not settings.LIVEKIT_API_KEY or not settings.LIVEKIT_API_SECRET:
            raise LiveKitConfigurationException(
                "LiveKit API credentials are not configured."
            )

    @staticmethod
    def _livekit_ttl(expires_at: datetime) -> timedelta:
        """
        Calculate the Time-To-Live for a LiveKit room token based on the session expiry.

        Args:
            expires_at: The absolute expiration datetime for the candidate session.

        Returns:
            A timedelta representing the remaining time (minimum 1 minute).
        """
        normalized = (
            expires_at
            if expires_at.tzinfo is not None
            else expires_at.replace(tzinfo=UTC)
        )
        remaining = normalized - datetime.now(UTC)
        return max(timedelta(minutes=1), remaining)

    @staticmethod
    def _datetime_value(value: object) -> datetime:
        """Normalize a datetime returned by the internal JSON API."""

        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise SessionTokenExpiryInvalidException(
                    "Candidate session expiry is invalid."
                ) from exc
        raise SessionTokenExpiryMissingException("Candidate session expiry is missing.")

    @staticmethod
    def _access_token(
        *,
        participant_identity: str,
        candidate_name: str,
        room_name: str,
        ttl: timedelta,
    ) -> api.AccessToken:
        """Build the shared candidate access-token grants."""

        return (
            api.AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
            .with_identity(participant_identity)
            .with_name(candidate_name)
            .with_ttl(ttl)
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
            )
        )

    @staticmethod
    async def _ensure_agent_dispatch(
        room_name: str,
        agent_metadata: dict[str, str],
    ) -> None:
        """
        Explicitly dispatch one interviewer for this connection generation.

        Token room configuration is only applied when LiveKit creates a room.
        A reconnect can join the previous room while its old agent job is still
        shutting down, so it needs an explicit dispatch even though the room
        already exists.
        """

        encoded_metadata = json.dumps(agent_metadata, sort_keys=True)
        livekit_api = api.LiveKitAPI(
            url=settings.LIVEKIT_URL,
            api_key=settings.LIVEKIT_API_KEY,
            api_secret=settings.LIVEKIT_API_SECRET,
        )
        try:
            try:
                dispatches = await livekit_api.agent_dispatch.list_dispatch(
                    room_name=room_name
                )
            except api.TwirpError as exc:
                if exc.status != 404:
                    raise
                # Explicit dispatch creates a missing room automatically.
                dispatches = []
            if any(
                dispatch.agent_name == settings.LIVEKIT_AGENT_NAME
                and dispatch.metadata == encoded_metadata
                for dispatch in dispatches
            ):
                return

            await livekit_api.agent_dispatch.create_dispatch(
                api.CreateAgentDispatchRequest(
                    agent_name=settings.LIVEKIT_AGENT_NAME,
                    room=room_name,
                    metadata=encoded_metadata,
                )
            )
        finally:
            await livekit_api.aclose()

    async def create_candidate_token(
        self,
        payload: LiveKitTokenRequest,
    ) -> LiveKitTokenResponse:
        """
        Validate a candidate session token and return LiveKit credentials for the interview room.

        Args:
            payload: The incoming token request containing the session token string.

        Returns:
            The authorized LiveKitTokenResponse containing the JWT and connection details.

        Raises:
            LiveKitConfigurationException: If LiveKit credentials are missing.
            SessionTokenExpiryInvalidException: If session expiry is invalid.
            SessionTokenExpiryMissingException: If session expiry is missing.
        """

        self._validate_configuration()
        context: CandidateConnectionContext = (
            await self._core_api_session_service.authorize_interview_connection(
                payload.session_token
            )
        )
        candidate_assessment_id = str(context.candidate_assessment_id)
        candidate_id = str(context.candidate_id)
        room_name = candidate_room_name(candidate_assessment_id)
        participant_identity = candidate_participant_identity(candidate_id)
        candidate_name = context.candidate_name

        agent_metadata = {
            "session_mode": "interview",
            "candidate_assessment_id": candidate_assessment_id,
            "candidate_id": candidate_id,
            "assessment_id": str(context.assessment_id),
            "interview_session_id": str(context.session_id),
            "connection_id": context.connection_id,
        }

        token = self._access_token(
            participant_identity=participant_identity,
            candidate_name=candidate_name,
            room_name=room_name,
            ttl=self._livekit_ttl(context.session_token_expires_at),
        ).to_jwt()
        await self._ensure_agent_dispatch(room_name, agent_metadata)

        return LiveKitTokenResponse(
            livekit_url=settings.LIVEKIT_URL,
            token=token,
            room_name=room_name,
            elapsed_secs=context.elapsed_secs,
            interview_started=context.interview_started,
        )

    async def create_demo_token(
        self,
        payload: LiveKitTokenRequest,
    ) -> LiveKitTokenResponse:
        """
        Issue credentials for an isolated, non-persistent practice room.

        The same LiveKit worker and speech pipeline are used, while job metadata
        explicitly routes response generation to the static demo agent.

        Args:
            payload: The incoming request with the session token.

        Returns:
            The authorized LiveKitTokenResponse for a newly generated demo room.
        """

        self._validate_configuration()
        context = await self._core_api_session_service.authorize_demo(
            payload.session_token
        )
        candidate_assessment_id = str(context["candidate_assessment_id"])
        candidate_id = str(context["candidate_id"])
        practice_session_id = uuid.uuid4().hex[:12]
        room_name = demo_room_name(
            candidate_assessment_id,
            practice_session_id,
        )
        participant_identity = candidate_participant_identity(candidate_id)
        candidate_name = str(context.get("candidate_name") or "Candidate")
        agent_metadata = {
            "session_mode": "demo",
            "practice_session_id": practice_session_id,
            "candidate_assessment_id": candidate_assessment_id,
            "candidate_id": candidate_id,
            "assessment_id": str(context.get("assessment_id") or ""),
        }

        token = (
            self._access_token(
                participant_identity=participant_identity,
                candidate_name=candidate_name,
                room_name=room_name,
                ttl=min(
                    timedelta(minutes=30),
                    self._livekit_ttl(
                        self._datetime_value(context["session_token_expires_at"])
                    ),
                ),
            )
            .with_room_config(
                api.RoomConfiguration(
                    agents=[
                        api.RoomAgentDispatch(
                            agent_name=settings.LIVEKIT_AGENT_NAME,
                            metadata=json.dumps(agent_metadata),
                        )
                    ]
                )
            )
            .to_jwt()
        )

        return LiveKitTokenResponse(
            livekit_url=settings.LIVEKIT_URL,
            token=token,
            room_name=room_name,
        )
