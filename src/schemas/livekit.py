"""Request and response schemas for LiveKit interview sessions."""

from __future__ import annotations

import uuid

from src.schemas.base import AppBaseModel


class LiveKitTokenRequest(AppBaseModel):
    """Request payload for candidate LiveKit room access."""

    invitation_token: uuid.UUID


class LiveKitTokenResponse(AppBaseModel):
    """Signed LiveKit room credentials returned to the candidate browser."""

    livekit_url: str
    token: str
    room_name: str
