"""Request and response schemas for LiveKit interview sessions."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import Field

from src.schemas.base import AppBaseModel


class LiveKitTokenRequest(AppBaseModel):
    """Request payload for session-authenticated LiveKit room access."""

    session_token: str = Field(min_length=32, max_length=255)


class CandidateSessionEntryRequest(AppBaseModel):
    """One-time invitation credential exchanged for a browser session token."""

    invitation_token: uuid.UUID


class CandidateSessionContextRequest(AppBaseModel):
    """Existing browser session credential used after refresh/reconnect."""

    session_token: str = Field(min_length=32, max_length=255)


class CandidateSessionBootstrapResponse(AppBaseModel):
    """Candidate waiting-room context plus the durable reconnect credential."""

    session_token: str
    session_token_expires_at: datetime
    session_status: str
    invite_reissued: bool
    interview_started: bool
    reconnect_deadline: datetime | None
    disconnect_count: int
    candidate_name: str
    company_name: str
    assessment_title: str
    interview_duration_mins: int
    window_end: datetime
    status: str
    sections_overview: list[str]


class CandidateConnectionContext(AppBaseModel):
    """Internal authorization result used to create a LiveKit access token."""

    session_id: uuid.UUID
    connection_id: str
    candidate_assessment_id: uuid.UUID
    candidate_id: uuid.UUID
    assessment_id: uuid.UUID
    candidate_name: str
    session_token_expires_at: datetime


class LiveKitTokenResponse(AppBaseModel):
    """Signed LiveKit room credentials returned to the candidate browser."""

    livekit_url: str
    token: str
    room_name: str
