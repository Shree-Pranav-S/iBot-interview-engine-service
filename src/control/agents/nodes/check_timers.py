"""Timer gate node for the interview router."""

from __future__ import annotations

from src.control.agents.state import InterviewState
from src.control.time_manager import (
    compute_elapsed,
    compute_section_time_remaining,
    total_allocated_secs,
)


async def check_timers(state: InterviewState) -> dict:
    elapsed = compute_elapsed(state)
    total_allocated_secs(state)
    section_remaining = int(compute_section_time_remaining(state))
    patch: dict = {
        "total_elapsed_secs": elapsed,
        "current_section_time_remaining_secs": max(0, section_remaining),
    }

    response_class = state.get("response_class")

    if response_class == "time_up":
        patch.update(
            {
                "auto_submit_triggered": True,
                "should_close": True,
                "session_status": "COMPLETED",
                "next_node": "closing",
            }
        )
    elif section_remaining <= 0 and response_class == "silence":
        patch["next_node"] = "section_transition"
    else:
        patch["next_node"] = "route_response"
    return patch
