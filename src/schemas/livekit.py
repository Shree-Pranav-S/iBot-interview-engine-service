"""Request and response schemas for LiveKit interview sessions."""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import ConfigDict, Field

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

    # This payload crosses a service boundary. Core API may add optional context
    # fields before the interview engine is rolled forward, so additive fields
    # must not make candidate authorization fail during a staggered deployment.
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    session_id: uuid.UUID
    connection_id: str
    candidate_assessment_id: uuid.UUID
    candidate_id: uuid.UUID
    assessment_id: uuid.UUID
    candidate_name: str
    session_token_expires_at: datetime
    elapsed_secs: int = 0
    interview_started: bool = False
    tab_switch_count: int = Field(default=0, ge=0)


class LiveKitTokenResponse(AppBaseModel):
    """Signed LiveKit room credentials returned to the candidate browser."""

    livekit_url: str
    token: str
    room_name: str
    elapsed_secs: int = 0
    interview_started: bool = False
    tab_switch_count: int = Field(default=0, ge=0)
