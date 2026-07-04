"""Typed contracts for durable cross-service event logging."""

import uuid
from enum import StrEnum
from typing import Any

from pydantic import Field

from src.schemas.base import AppBaseModel


class EventName(StrEnum):
    """Durable event names emitted by this service."""

    INTERVIEW_STARTED = "INTERVIEW_STARTED"
    INTERVIEW_ENDED = "INTERVIEW_ENDED"
    CELERY_TASK_STARTED = "CELERY_TASK_STARTED"
    CELERY_TASK_RETRYING = "CELERY_TASK_RETRYING"
    CELERY_TASK_COMPLETED = "CELERY_TASK_COMPLETED"
    CELERY_TASK_FAILED = "CELERY_TASK_FAILED"


class EventSource(StrEnum):
    """Source label used for interview-engine events."""

    INTERVIEW_ENGINE = "INTERVIEW_ENGINE"


class EventLogCreate(AppBaseModel):
    """Payload for creating one durable event-log record."""

    event_name: EventName
    source_service: EventSource
    correlation_id: str = Field(min_length=1, max_length=255)
    candidate_assessment_id: uuid.UUID | None = None
    recruiter_id: uuid.UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
