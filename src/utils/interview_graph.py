"""Helpers shared by the LangGraph interview workflow."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp for durable JSON records."""

    return datetime.now(UTC).isoformat()


def deterministic_turn_id(
    session_id: str,
    turn_number: int,
    speaker: str,
) -> str:
    """Build the deterministic transcript turn id required for retries."""

    return f"{session_id}:{int(turn_number)}:{speaker}"


def deterministic_violation_id(
    session_id: str,
    turn_number: int,
    violation_type: str,
) -> str:
    """Build the deterministic violation id required for retries."""

    return f"{session_id}:{int(turn_number)}:{violation_type}"


def append_recent_turn(
    recent_turns: list[dict[str, Any]] | None,
    turn: dict[str, Any],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Append a turn to bounded graph memory without duplicating turn_id."""

    current = list(recent_turns or [])
    turn_id = str(turn.get("turn_id") or "")
    if turn_id and any(str(item.get("turn_id") or "") == turn_id for item in current):
        return current[-limit:]
    return [*current, turn][-limit:]


def clean_section_name(value: Any) -> str:
    """Normalize a section name for state keys and question ids."""

    text = str(value or "general").strip() or "general"
    return "_".join(text.lower().split())
