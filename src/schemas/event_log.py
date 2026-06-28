"""Typed contracts for durable cross-service event logging."""

import uuid
from enum import StrEnum
from typing import Any

from pydantic import Field

from src.schemas.base import AppBaseModel


class EventName(StrEnum):
    INVITE_LINK_CONSUMED = "INVITE_LINK_CONSUMED"
    SESSION_TOKEN_REISSUED = "SESSION_TOKEN_REISSUED"
    INVITE_LINK_REJECTED = "INVITE_LINK_REJECTED"
    SESSION_DISCONNECTED = "SESSION_DISCONNECTED"
    SESSION_TERMINATED_MAX_DISCONNECTS = "SESSION_TERMINATED_MAX_DISCONNECTS"
    SESSION_RECONNECTED = "SESSION_RECONNECTED"
    SESSION_DISCONNECT_TIMEOUT_RECORDED = "SESSION_DISCONNECT_TIMEOUT_RECORDED"
    SESSION_TERMINATED_RECONNECT_TIMEOUT = "SESSION_TERMINATED_RECONNECT_TIMEOUT"
    INTERVIEW_STARTED = "INTERVIEW_STARTED"
    INTERVIEW_DISCONNECTED = "INTERVIEW_DISCONNECTED"
    INTERVIEW_RECONNECTED = "INTERVIEW_RECONNECTED"
    INTERVIEW_ENDED = "INTERVIEW_ENDED"
    CSV_ROW_FAILED = "CSV_ROW_FAILED"
    CELERY_TASK_STARTED = "CELERY_TASK_STARTED"
    CELERY_TASK_RETRYING = "CELERY_TASK_RETRYING"
    CELERY_TASK_COMPLETED = "CELERY_TASK_COMPLETED"
    CELERY_TASK_FAILED = "CELERY_TASK_FAILED"


class EventSource(StrEnum):
    CORE_API = "CORE_API"
    INTERVIEW_ENGINE = "INTERVIEW_ENGINE"


class EventLogCreate(AppBaseModel):
    event_name: EventName
    source_service: EventSource
    correlation_id: str = Field(min_length=1, max_length=255)
    candidate_assessment_id: uuid.UUID | None = None
    recruiter_id: uuid.UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
