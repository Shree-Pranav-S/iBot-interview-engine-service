"""Typed recruiter dashboard events shared through Redis pub/sub."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class RecruiterEventType(StrEnum):
    """Recruiter dashboard event types published by this service."""

    INTERVIEW_EVALUATED = "INTERVIEW_EVALUATED"


class RecruiterRealtimeEvent(BaseModel):
    """Redis payload delivered to a recruiter's dashboard channel."""

    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    event_type: RecruiterEventType
    occurred_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)
