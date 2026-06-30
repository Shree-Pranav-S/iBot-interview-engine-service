"""Deterministic API-key affinity for live interview turns."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def compute_turn_key_slot(state: Mapping[str, Any]) -> int:
    """Return a stable, session-offset slot seed for the current candidate turn."""

    session_id = str(
        state.get("interview_session_id")
        or state.get("candidate_assessment_id")
        or "interview"
    )
    turn_number = max(1, int(state.get("turn_number") or 1))
    digest = hashlib.sha256(session_id.encode("utf-8")).digest()
    session_offset = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return session_offset + turn_number - 1


def active_turn_key_slot(state: Mapping[str, Any]) -> int:
    """Use the slot captured for this turn, falling back to a fresh calculation."""

    captured = state.get("llm_key_slot")
    if captured is not None:
        return int(captured)
    return compute_turn_key_slot(state)
