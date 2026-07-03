"""Answer-boundary timing and forced section-overrun transitions."""

from __future__ import annotations

from typing import Any

from src.control.agents.nodes.persist_turn import (
    persist_candidate_output,
    schedule_elapsed_persistence,
)
from src.control.agents.nodes.question_strategy import current_section_name
from src.control.agents.state import InterviewState
from src.control.agents.templates import choose_template

SECTION_TRANSITION_THRESHOLD_SECS = 10
SECTION_BARGE_IN_GRACE_SECS = 30
FORCE_BEHAVIOURAL_REMAINING_SECS = 20


def _elapsed_from_event(state: InterviewState) -> int:
    """
    Determine the total elapsed time using the timestamp embedded in the latest event.

    Args:
        state: The current interview state.

    Returns:
        The elapsed time in seconds.
    """
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
    """Return the section deadline, including any requested overrun grace."""

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
        total_duration - future_reserved + grace
        if future_reserved > 0
        else total_duration + grace
    )
    return max(0, min(nominal_deadline, protected_deadline))


def _timing(state: InterviewState) -> dict[str, int]:
    """
    Calculate all derived timing properties for the current interview state.

    Args:
        state: The current interview state.

    Returns:
        A dictionary of various calculated timing values in seconds.
    """
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
    """
    Map calculated timing values to the formal keys expected in InterviewState.

    Args:
        timing: The dictionary returned by `_timing`.

    Returns:
        A dictionary of state updates.
    """
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
    """
    Find the next valid section (technical or behavioural) in the runtime plan.

    Args:
        state: The current interview state.

    Returns:
        A tuple of (section_index, section_dict), or None if no sections remain.
    """
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


def _behavioural_rescue_section(
    state: InterviewState,
) -> tuple[int, dict[str, Any]] | None:
    """Return behavioural section when it is still ahead of current index."""

    sections = list(state.get("runtime_sections") or [])
    current = int(state.get("current_section_index") or 0)
    for index, section in enumerate(sections):
        if section.get("section_kind") == "behavioural_cultural" and index > current:
            return index, section
    return None


def _section_label(section: dict[str, Any] | None, fallback: str) -> str:
    """
    Generate a human-readable label for a section, preferring the skill name.

    Args:
        section: The section dictionary, or None.
        fallback: The fallback string if the section has no explicit label.

    Returns:
        A clean, space-separated label string.
    """
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
    """
    Construct the state updates required to route the graph into the closing node.

    Args:
        timing: The computed timing dictionary.
        reason: The reason code for closing.

    Returns:
        A dictionary of state updates.
    """
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
    """
    Construct the state updates required to transition to the next interview section.

    Args:
        state: The current interview state.
        timing: The computed timing dictionary.
        next_index: The index of the next section.
        next_section: The next section dictionary.
        reason: The reason for the transition.
        template_name: The name of the preface template to use.
        barge_in: Whether this was a forced transition due to timeout.

    Returns:
        A dictionary of state updates.
    """
    current_label = _section_label(
        None,
        current_section_name(state),
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


def decide_time_action(state: InterviewState) -> dict[str, Any]:
    """
    Decide what should follow a (hypothetically) substantial answer, without an LLM.

    Pure helper so the merged interviewer-turn node can resolve the target
    section before asking the model to generate the next question. The decision uses
    only elapsed time and section budgets, so it is identical whether computed before
    or after classification.

    Args:
        state: The current interview state.

    Returns:
        A decision dictionary with the action ("continue", "transition", or
        "close"), the mapped timing updates, the resolved next section, and labels.
    """

    timing = _timing(state)
    timing_updates = _timing_updates(timing)
    next_section = _next_section(state)
    current_label = _section_label(
        None,
        current_section_name(state),
    )

    def _close(reason: str) -> dict[str, Any]:
        return {
            "action": "close",
            "timing_updates": timing_updates,
            "pending_section_index": None,
            "suppress_previous_context_for_next_question": False,
            "transition_reason": None,
            "closing_reason": reason,
            "current_label": current_label,
            "next_label": None,
            "next_index": None,
            "next_section": None,
        }

    def _transition_decision(
        *,
        next_index: int,
        section: dict[str, Any],
        reason: str,
    ) -> dict[str, Any]:
        return {
            "action": "transition",
            "timing_updates": timing_updates,
            "pending_section_index": next_index,
            "suppress_previous_context_for_next_question": True,
            "transition_reason": reason,
            "closing_reason": None,
            "current_label": current_label,
            "next_label": _section_label(section, "the next section"),
            "next_index": next_index,
            "next_section": section,
        }

    if timing["remaining_secs"] <= 0:
        return _close("interview_time_exhausted")

    behavioural_rescue = _behavioural_rescue_section(state)
    if (
        behavioural_rescue is not None
        and timing["remaining_secs"] <= FORCE_BEHAVIOURAL_REMAINING_SECS
    ):
        next_index, section = behavioural_rescue
        return _transition_decision(
            next_index=next_index,
            section=section,
            reason="behavioural_time_rescue",
        )

    if state.get("current_section_kind") == "self_intro":
        if next_section is None:
            return _close("interview_plan_complete")
        next_index, section = next_section
        return _transition_decision(
            next_index=next_index,
            section=section,
            reason="self_introduction_complete",
        )

    if (
        next_section is None
        and timing["remaining_secs"] <= SECTION_TRANSITION_THRESHOLD_SECS
    ):
        return _close("interview_time_exhausted")

    section_time_low = (
        timing["section_elapsed_secs"] >= timing["section_budget_secs"]
        or timing["section_remaining_secs"] <= SECTION_TRANSITION_THRESHOLD_SECS
    )
    if section_time_low:
        if next_section is None:
            return _close("final_section_complete")
        next_index, section = next_section
        return _transition_decision(
            next_index=next_index,
            section=section,
            reason="section_time_exhausted",
        )

    return {
        "action": "continue",
        "timing_updates": timing_updates,
        "pending_section_index": None,
        "suppress_previous_context_for_next_question": bool(
            state.get("suppress_previous_context_for_next_question")
        ),
        "transition_reason": None,
        "closing_reason": None,
        "current_label": current_label,
        "next_label": current_label,
        "next_index": None,
        "next_section": None,
    }


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

    behavioural_rescue = _behavioural_rescue_section(state)
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
