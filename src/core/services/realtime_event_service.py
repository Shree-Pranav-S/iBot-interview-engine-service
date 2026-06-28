"""Publish recruiter-scoped evaluation completion events."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from src.data.clients.redis_client import get_or_create_client
from src.schemas.realtime import RecruiterEventType, RecruiterRealtimeEvent

logger = logging.getLogger(__name__)
RECRUITER_CHANNEL_PREFIX = "recruiter-dashboard"


async def publish_recruiter_event(
    *,
    recruiter_id: uuid.UUID | str,
    event_type: RecruiterEventType,
    payload: dict[str, Any],
) -> bool:
    """Publish after commit without making Redis availability fail evaluation."""

    event = RecruiterRealtimeEvent(
        event_type=event_type,
        occurred_at=datetime.now(UTC),
        payload=payload,
    )
    try:
        redis_client = await get_or_create_client()
        await redis_client.publish(
            f"{RECRUITER_CHANNEL_PREFIX}:{recruiter_id}",
            event.model_dump_json(),
        )
        logger.info(
            "Published recruiter dashboard event",
            extra={
                "recruiter_id": str(recruiter_id),
                "event_id": str(event.event_id),
                "event_type": event_type.value,
            },
        )
        return True
    except Exception:
        logger.exception(
            "Could not publish recruiter dashboard event",
            extra={
                "recruiter_id": str(recruiter_id),
                "event_type": event_type.value,
            },
        )
        return False
