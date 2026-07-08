"""Helpers for normalizing LiveKit resume events into graph candidate events."""

from __future__ import annotations

from typing import Any


def _normalize_event(raw: Any) -> dict[str, Any]:
    """
    Ensure the incoming raw event from LiveKit is formatted as a consistent dictionary.

    Args:
        raw: The raw event which may be a string (like "__SILENCE__") or a dict.

    Returns:
        A normalized dictionary with at least an 'event_type' and 'text'.
    """
    if isinstance(raw, str):
        if raw == "__SILENCE__":
            return {
                "event_type": "silence_timeout",
                "text": "",
                "silence_duration_ms": 5000,
            }
        return {"event_type": "candidate_answer", "text": raw}

    if isinstance(raw, dict):
        return dict(raw)
    return {"event_type": "candidate_answer", "text": ""}
