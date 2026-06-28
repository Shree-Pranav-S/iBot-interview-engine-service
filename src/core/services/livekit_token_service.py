"""Business logic for issuing LiveKit interview room tokens."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

from livekit import api

from src.config.settings import settings
from src.core.exceptions import InternalServerException
from src.core.services.candidate_session_service import (
    CandidateSessionService,
)
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
        candidate_session_service: CandidateSessionService | None = None,
    ) -> None:
        self._candidate_session_service = (
            candidate_session_service or CandidateSessionService()
        )

    @staticmethod
    def _validate_configuration() -> None:
        if not settings.LIVEKIT_URL:
            raise InternalServerException("LiveKit URL is not configured.")
        if not settings.LIVEKIT_API_KEY or not settings.LIVEKIT_API_SECRET:
            raise InternalServerException("LiveKit API credentials are not configured.")

    @staticmethod
    def _livekit_ttl(expires_at: datetime) -> timedelta:
        normalized = (
            expires_at
            if expires_at.tzinfo is not None
            else expires_at.replace(tzinfo=UTC)
        )
        remaining = normalized - datetime.now(UTC)
        return max(timedelta(minutes=1), remaining)

    async def create_candidate_token(
        self,
        payload: LiveKitTokenRequest,
    ) -> LiveKitTokenResponse:
        """Validate a candidate session token and return LiveKit credentials."""

        self._validate_configuration()
        context: CandidateConnectionContext = (
            await self._candidate_session_service.authorize_interview_connection(
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

        token = (
            api.AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
            .with_identity(participant_identity)
            .with_name(candidate_name)
            .with_ttl(self._livekit_ttl(context.session_token_expires_at))
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
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

    async def create_demo_token(
        self,
        payload: LiveKitTokenRequest,
    ) -> LiveKitTokenResponse:
        """
        Issue credentials for an isolated, non-persistent practice room.

        The same LiveKit worker and speech pipeline are used, while job metadata
        explicitly routes response generation to the static demo agent.
        """

        self._validate_configuration()
        context = await self._candidate_session_service.authorize_demo(
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
            api.AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
            .with_identity(participant_identity)
            .with_name(candidate_name)
            .with_ttl(
                min(
                    timedelta(minutes=30),
                    self._livekit_ttl(context["session_token_expires_at"]),
                )
            )
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                    can_publish=True,
                    can_subscribe=True,
                    can_publish_data=True,
                )
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
