"""Helpers for extracting graph turn output."""

from __future__ import annotations

from typing import Any


def new_bot_turns(
    previous_state: dict[str, Any] | None,
    state: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return bot turns visible in bounded graph memory."""

    previous_ids = {
        str(turn.get("turn_id") or "")
        for turn in (previous_state or {}).get("recent_turns", [])
    }
    return [
        turn
        for turn in state.get("recent_turns", [])
        if turn.get("speaker") == "bot"
        and turn.get("text")
        and str(turn.get("turn_id") or "") not in previous_ids
    ]


def extract_bot_reply(
    previous_state: dict[str, Any] | None,
    state: dict[str, Any],
) -> str:
    """Return new bot output as one spoken text response."""

    direct_reply = str(state.get("bot_reply_text") or "").strip()
    previous_direct_reply = str(
        (previous_state or {}).get("bot_reply_text") or ""
    ).strip()
    if direct_reply and direct_reply != previous_direct_reply:
        return direct_reply

    return " ".join(
        str(turn.get("text") or "").strip()
        for turn in new_bot_turns(previous_state, state)
        if turn.get("text")
    ).strip()
