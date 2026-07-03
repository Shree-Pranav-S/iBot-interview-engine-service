"""Helpers shared by the LangGraph interview workflow."""

from __future__ import annotations

from datetime import UTC, datetime


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
