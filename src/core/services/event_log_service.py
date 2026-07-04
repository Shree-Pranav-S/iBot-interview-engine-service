"""Business service for durable event recording."""

import logging

from src.clients.core_api_client import get_core_api_client
from src.schemas.event_log import EventLogCreate

logger = logging.getLogger(__name__)


async def try_record_event_in_background(event: EventLogCreate) -> None:
    """Best-effort logging that never changes the interview or task outcome."""

    try:
        await get_core_api_client().create_event_log(event)
    except Exception:
        logger.exception(
            "Failed to persist event log",
            extra={
                "event_name": event.event_name.value,
                "correlation_id": event.correlation_id,
            },
        )
