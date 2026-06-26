"""Business logic for issuing LiveKit interview room tokens."""

from __future__ import annotations

import json
from datetime import timedelta

from livekit import api

from src.config.settings import settings
from src.core.exceptions import (
    AuthenticationException,
    ForbiddenException,
    InternalServerException,
)
from src.data.repositories import interview_workflow_repository as db
from src.schemas.livekit import LiveKitTokenRequest, LiveKitTokenResponse
from src.utils.livekit import candidate_participant_identity, candidate_room_name


class LiveKitTokenService:
    """Service layer for candidate LiveKit room credentials."""

    async def create_candidate_token(
        self,
        payload: LiveKitTokenRequest,
    ) -> LiveKitTokenResponse:
        """Validate invitation token and return signed LiveKit room credentials."""

        context = await db.load_candidate_livekit_context(payload.invitation_token)
        if not context:
            raise AuthenticationException("Invalid or expired invitation token.")

        candidate_assessment_id = str(context["candidate_assessment_id"])
        can_start, reason = await db.assert_session_can_start(candidate_assessment_id)
        if not can_start:
            raise ForbiddenException(reason or "Interview session cannot be started.")

        if not settings.LIVEKIT_URL:
            raise InternalServerException("LiveKit URL is not configured.")
        if not settings.LIVEKIT_API_KEY or not settings.LIVEKIT_API_SECRET:
            raise InternalServerException("LiveKit API credentials are not configured.")

        candidate_id = str(context["candidate_id"])
        room_name = candidate_room_name(candidate_assessment_id)
        participant_identity = candidate_participant_identity(candidate_id)
        candidate_name = str(context.get("candidate_name") or "Candidate")

        agent_metadata = {
            "candidate_assessment_id": candidate_assessment_id,
            "candidate_id": candidate_id,
            "assessment_id": str(context.get("assessment_id") or ""),
        }

        token = (
            api.AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
            .with_identity(participant_identity)
            .with_name(candidate_name)
            .with_ttl(timedelta(hours=2))
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
