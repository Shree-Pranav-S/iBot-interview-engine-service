"""Simple time-budget checks for the interview graph."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from src.control.agents.nodes.context_utils import (
    is_behavioural_cultural_section,
)
from src.control.agents.state import InterviewState
from src.data.repositories import interview_session_repository

logger = logging.getLogger(__name__)
SOFT_OVERRUN_GRACE_SECS = 0


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _elapsed_since(value: str | None) -> int:
    parsed = _parse_dt(value)
    if parsed is None:
        return 0
    return max(0, int((datetime.now(UTC) - parsed).total_seconds()))


def _has_next_section(state: InterviewState) -> bool:
    section_order = state.get("section_order") or []
    return int(state.get("current_section_index") or 0) < len(section_order) - 1


def current_section_overrun_secs(state: InterviewState) -> int:
    section = state.get("current_section") or "general"
    section_budget = int((state.get("section_budgets") or {}).get(section) or 0)
    if not section_budget:
        return 0
    section_elapsed = _elapsed_since(state.get("section_started_at"))
    if not section_elapsed:
        section_elapsed = int(state.get("current_section_elapsed_secs") or 0)
    return max(0, section_elapsed - section_budget)


async def check_time_budget(state: InterviewState) -> dict[str, object]:
    started_at = _parse_dt(state.get("started_at"))
    total_pause_secs = int(state.get("total_pause_secs") or 0)
    if started_at:
        elapsed_secs = max(
            0,
            int((datetime.now(UTC) - started_at).total_seconds()) - total_pause_secs,
        )
    else:
        elapsed_secs = int(state.get("elapsed_secs") or 0)

    total_duration_secs = int(state.get("total_duration_secs") or 0)
    remaining_secs = max(0, total_duration_secs - elapsed_secs)
    total_overrun = max(0, elapsed_secs - total_duration_secs)

    section = state.get("current_section") or "general"
    section_budget = int((state.get("section_budgets") or {}).get(section) or 0)
    section_elapsed = _elapsed_since(state.get("section_started_at"))
    if not section_elapsed:
        section_elapsed = int(state.get("current_section_elapsed_secs") or 0)
    section_remaining = (
        max(0, section_budget - section_elapsed) if section_budget else remaining_secs
    )
    section_overrun = max(0, section_elapsed - section_budget) if section_budget else 0

    response_completed = bool(state.get("last_candidate_event"))
    next_action = state.get("next_action") or "continue"
    behavioural_budget_exhausted = (
        is_behavioural_cultural_section(state)
        and bool(section_budget)
        and section_elapsed >= section_budget
        and response_completed
    )
    force_close = (
        remaining_secs <= 0 or behavioural_budget_exhausted
    ) and response_completed
    force_transition = (
        section_overrun > 0 and response_completed and _has_next_section(state)
    )
    force_behavioural_cultural = False
    next_section_index = state.get("next_section_index")
    closing_reason = state.get("closing_reason")
    if force_close or state.get("last_response_type") == "timer_expired":
        next_action = "complete"
        closing_reason = closing_reason or (
            "behavioural_cultural_budget_exhausted"
            if behavioural_budget_exhausted
            else "time_expired"
        )
    elif force_transition:
        next_action = "section_transition"
        next_section_index = int(state.get("current_section_index") or 0) + 1
        closing_reason = None
    elif next_action not in {"complete", "section_transition", "next_question"}:
        next_action = "next_question"

    session_id = state.get("interview_session_id")
    if session_id:
        await interview_session_repository.update_elapsed_time(
            session_id,
            elapsed_secs=elapsed_secs,
            total_pause_secs=total_pause_secs,
        )

    logger.info(
        "checked interview time budget",
        extra={
            "candidate_assessment_id": state.get("candidate_assessment_id"),
            "elapsed_secs": elapsed_secs,
            "remaining_secs": remaining_secs,
            "section": section,
            "section_remaining_secs": section_remaining,
            "next_action": next_action,
        },
    )

    return {
        "elapsed_secs": elapsed_secs,
        "remaining_secs": remaining_secs,
        "current_section_elapsed_secs": section_elapsed,
        "current_section_remaining_secs": section_remaining,
        "soft_total_overrun_secs": total_overrun,
        "soft_section_overrun_secs": section_overrun,
        "force_close_due_to_overrun": force_close,
        "force_transition_due_to_overrun": force_transition,
        "force_behavioural_cultural_due_to_time": force_behavioural_cultural,
        "close_after_behavioural_cultural": bool(
            state.get("close_after_behavioural_cultural") or force_behavioural_cultural
        ),
        "next_action": next_action,
        "next_section_index": next_section_index,
        "closing_reason": closing_reason,
        "should_close": next_action == "complete",
    }
