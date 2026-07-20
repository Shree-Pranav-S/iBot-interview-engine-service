"""Request and response schemas for LiveKit interview sessions."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

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


class FaceProctoringPacket(AppBaseModel):
    """Duration-qualified browser face-presence/count observation episode."""

    type: Literal["face_absent", "multiple_faces"]
    event_id: uuid.UUID
    condition_started_at: datetime
    observed_duration_ms: int = Field(ge=3_000, le=3_600_000)
    sample_count: int = Field(ge=1, le=10_000)
    max_face_count: int = Field(ge=0, le=20)
    min_confidence: float | None = Field(default=None, ge=0, le=1)
    max_confidence: float | None = Field(default=None, ge=0, le=1)
    source: Literal["mediapipe_face_detector", "camera_state"]
    detector_version: str = Field(min_length=1, max_length=100)
    model_name: str = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def validate_policy_claim(self) -> Self:
        """Reject packets that do not meet the public proctoring policy."""

        minimum_duration = 5_000 if self.type == "face_absent" else 3_000
        if self.observed_duration_ms < minimum_duration:
            raise ValueError(
                f"{self.type} must persist for at least {minimum_duration}ms"
            )
        if self.source == "camera_state" and self.type != "face_absent":
            raise ValueError("camera_state can only report face_absent")
        if self.type == "face_absent" and self.max_face_count != 0:
            raise ValueError("face_absent cannot report a positive face count")
        if self.type == "multiple_faces" and self.max_face_count < 2:
            raise ValueError("multiple_faces must report at least two faces")
        if (
            self.min_confidence is not None
            and self.max_confidence is not None
            and self.min_confidence > self.max_confidence
        ):
            raise ValueError("min_confidence cannot exceed max_confidence")
        return self
