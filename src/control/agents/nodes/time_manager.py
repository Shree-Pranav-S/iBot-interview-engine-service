"""Answer-boundary timing and forced section-overrun transitions."""

from __future__ import annotations

from typing import Any

from src.control.agents.nodes.persist_turn import persist_candidate_output
from src.control.agents.state import InterviewState
from src.control.agents.utils.persist_turn import schedule_elapsed_persistence
from src.control.agents.utils.time_manager import (
    FORCE_BEHAVIOURAL_REMAINING_SECS,
    SECTION_BARGE_IN_GRACE_SECS,
    _closing,
    _next_section,
    _timing,
    _timing_updates,
    _transition,
    section_transition_deadline_elapsed,
)

__all__ = [
    "FORCE_BEHAVIOURAL_REMAINING_SECS",
    "SECTION_BARGE_IN_GRACE_SECS",
    "force_section_time_barge_in",
    "section_transition_deadline_elapsed",
]


async def force_section_time_barge_in(
    state: InterviewState,
) -> dict[str, Any]:
    """
    Force movement on the event loop after the section grace period expires.

    This node is triggered when the candidate has been talking too long and
    the section has run out of time. It immediately forces a transition or closure.

    Args:
        state: The current interview state.

    Returns:
        State updates containing the barge-in routing logic and timing values.
    """

    timing = _timing(state)
    schedule_elapsed_persistence(
        state,
        elapsed_secs=timing["elapsed_secs"],
    )
    if timing["remaining_secs"] <= 0:
        persistence_updates = await persist_candidate_output(state)
        return {
            **_closing(timing, reason="interview_time_exhausted"),
            **persistence_updates,
            "barge_in_triggered": True,
        }

    if state.get("current_section_kind") == "self_intro":
        # A self-introduction transitions only after LiveKit confirms the completed
        # candidate turn and the deterministic substantiality rule accepts it.
        # Ignore stale/premature section watchdog events while total time remains.
        return {
            **_timing_updates(timing),
            "pending_candidate_turn": None,
            "barge_in_triggered": False,
            "next_action": "await_candidate_response",
        }

    behavioural_rescue = _next_section(
        state,
        section_kind="behavioural_cultural",
    )
    if (
        behavioural_rescue is not None
        and timing["remaining_secs"] <= FORCE_BEHAVIOURAL_REMAINING_SECS
    ):
        persistence_updates = await persist_candidate_output(state)
        next_index, section = behavioural_rescue
        return {
            **_transition(
                state,
                timing,
                next_index=next_index,
                next_section=section,
                reason="behavioural_time_rescue",
                template_name="behavioural_forced_transition",
                barge_in=True,
            ),
            **persistence_updates,
        }

    section_overrun = timing["elapsed_secs"] >= timing["section_barge_deadline_secs"]
    if not section_overrun:
        return {
            **_timing_updates(timing),
            "next_action": "await_candidate_response",
        }

    persistence_updates = await persist_candidate_output(state)
    next_section = _next_section(state)
    if next_section is None:
        return {
            **_closing(timing, reason="final_section_overrun"),
            **persistence_updates,
            "barge_in_triggered": True,
        }

    next_index, section = next_section
    return {
        **_transition(
            state,
            timing,
            next_index=next_index,
            next_section=section,
            reason="section_overrun_barge_in",
            template_name="barge_in_transition",
            barge_in=True,
        ),
        # The previous bot question had no completed candidate turn.
        **persistence_updates,
    }
