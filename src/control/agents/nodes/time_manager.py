"""Answer-boundary timing and forced section-overrun transitions."""

from __future__ import annotations

from typing import Any

from src.control.agents.nodes.persist_turn import (
    persist_candidate_output,
    schedule_elapsed_persistence,
)
from src.control.agents.state import InterviewState
from src.control.agents.templates import choose_template

SECTION_TRANSITION_THRESHOLD_SECS = 10
SECTION_BARGE_IN_GRACE_SECS = 30


def _elapsed_from_event(state: InterviewState) -> int:
    event = state.get("candidate_event")
    if isinstance(event, dict) and event.get("elapsed_secs") is not None:
        try:
            return max(0, int(event["elapsed_secs"]))
        except (TypeError, ValueError):
            pass
    return max(0, int(state.get("elapsed_secs") or 0))


def remaining_future_budget_secs(state: InterviewState) -> int:
    """Return time that must remain protected for sections after the current one."""

    current_index = int(state.get("current_section_index") or 0)
    sections = list(state.get("runtime_sections") or [])
    budgets = state.get("section_budgets_secs") or {}
    return sum(
        max(0, int(budgets.get(str(index)) or 0))
        for index, section in enumerate(sections)
        if index > current_index
        and section.get("section_kind")
        in {
            "technical",
            "behavioural_cultural",
        }
    )


def section_transition_deadline_elapsed(
    state: InterviewState,
    *,
    grace_secs: int = 0,
) -> int:
    """Return the section deadline without consuming future allocations."""

    grace = max(0, int(grace_secs))
    section_started = max(
        0,
        int(state.get("current_section_started_elapsed_secs") or 0),
    )
    section_budget = max(
        1,
        int(state.get("current_section_budget_secs") or 1),
    )
    nominal_deadline = section_started + section_budget + grace
    total_duration = max(1, int(state.get("total_duration_secs") or 1))
    future_reserved = remaining_future_budget_secs(state)
    protected_deadline = (
        total_duration - future_reserved
        if future_reserved > 0
        else total_duration + grace
    )
    return max(0, min(nominal_deadline, protected_deadline))


def _timing(state: InterviewState) -> dict[str, int]:
    elapsed = _elapsed_from_event(state)
    total = max(1, int(state.get("total_duration_secs") or 1))
    section_budget = max(
        1,
        int(state.get("current_section_budget_secs") or 1),
    )
    section_started = max(
        0,
        int(state.get("current_section_started_elapsed_secs") or 0),
    )
    section_elapsed = max(0, elapsed - section_started)
    future_reserved = remaining_future_budget_secs(state)
    transition_deadline = section_transition_deadline_elapsed(state)
    barge_deadline = section_transition_deadline_elapsed(
        state,
        grace_secs=SECTION_BARGE_IN_GRACE_SECS,
    )
    nominal_section_remaining = max(
        0,
        section_budget - section_elapsed,
    )
    scheduled_section_remaining = max(0, transition_deadline - elapsed)
    return {
        "elapsed_secs": elapsed,
        "remaining_secs": max(0, total - elapsed),
        "section_elapsed_secs": section_elapsed,
        "section_remaining_secs": min(
            nominal_section_remaining,
            scheduled_section_remaining,
        ),
        "nominal_section_remaining_secs": nominal_section_remaining,
        "section_budget_secs": section_budget,
        "future_reserved_secs": future_reserved,
        "section_transition_deadline_secs": transition_deadline,
        "section_barge_deadline_secs": barge_deadline,
    }


def _timing_updates(timing: dict[str, int]) -> dict[str, int]:
    return {
        "elapsed_secs": timing["elapsed_secs"],
        "remaining_secs": timing["remaining_secs"],
        "current_section_budget_secs": timing["section_budget_secs"],
        "current_section_elapsed_secs": timing["section_elapsed_secs"],
        "current_section_remaining_secs": timing["section_remaining_secs"],
        "reserved_future_section_secs": timing["future_reserved_secs"],
        "current_section_transition_deadline_elapsed_secs": timing[
            "section_transition_deadline_secs"
        ],
    }


def _next_section(
    state: InterviewState,
) -> tuple[int, dict[str, Any]] | None:
    sections = list(state.get("runtime_sections") or [])
    current = int(state.get("current_section_index") or 0)
    for index in range(current + 1, len(sections)):
        section = sections[index]
        if section.get("section_kind") in {
            "technical",
            "behavioural_cultural",
        }:
            return index, section
    return None


def _section_label(section: dict[str, Any] | None, fallback: str) -> str:
    if not section:
        return fallback.replace("_", " ")
    return str(section.get("skill") or section.get("section_name") or fallback).replace(
        "_", " "
    )


def _closing(
    timing: dict[str, int],
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        **_timing_updates(timing),
        "next_action": "generate_closing",
        "should_close": True,
        "closing_reason": reason,
        "pending_section_index": None,
        "response_preface_text": None,
    }


def _transition(
    state: InterviewState,
    timing: dict[str, int],
    *,
    next_index: int,
    next_section: dict[str, Any],
    reason: str,
    template_name: str,
    barge_in: bool,
) -> dict[str, Any]:
    current_label = _section_label(
        None,
        str(state.get("current_section") or "this section"),
    )
    next_label = _section_label(next_section, "the next section")
    preface = choose_template(
        template_name,
        current_section=current_label,
        next_section=next_label,
    )
    return {
        **_timing_updates(timing),
        "pending_section_index": next_index,
        "suppress_previous_context_for_next_question": True,
        "transition_reason": reason,
        "response_preface_text": preface,
        "barge_in_triggered": barge_in,
        "should_advance_question": True,
        "should_close": False,
        "next_action": "generate_question",
    }


async def check_time_after_substantial_answer(
    state: InterviewState,
) -> dict[str, Any]:
    """Check timing on the event loop after a substantial candidate answer."""

    timing = _timing(state)
    schedule_elapsed_persistence(
        state,
        elapsed_secs=timing["elapsed_secs"],
    )

    if timing["remaining_secs"] <= 0:
        return _closing(timing, reason="interview_time_exhausted")

    next_section = _next_section(state)
    if state.get("current_section_kind") == "self_intro":
        if next_section is None:
            return _closing(timing, reason="interview_plan_complete")
        next_index, section = next_section
        return _transition(
            state,
            timing,
            next_index=next_index,
            next_section=section,
            reason="self_introduction_complete",
            template_name="section_transition",
            barge_in=False,
        )

    if (
        next_section is None
        and timing["remaining_secs"] <= SECTION_TRANSITION_THRESHOLD_SECS
    ):
        return _closing(timing, reason="interview_time_exhausted")

    section_time_low = (
        timing["section_elapsed_secs"] >= timing["section_budget_secs"]
        or timing["section_remaining_secs"] <= SECTION_TRANSITION_THRESHOLD_SECS
    )
    if section_time_low:
        if next_section is None:
            return _closing(timing, reason="final_section_complete")
        next_index, section = next_section
        return _transition(
            state,
            timing,
            next_index=next_index,
            next_section=section,
            reason="section_time_exhausted",
            template_name="timed_section_transition",
            barge_in=False,
        )

    return {
        **_timing_updates(timing),
        "next_action": "generate_question",
        "should_close": False,
        "closing_reason": None,
    }


async def force_section_time_barge_in(
    state: InterviewState,
) -> dict[str, Any]:
    """Force movement on the event loop after the section grace period."""

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
