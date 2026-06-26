"""Session initialization node for the interview graph."""

from __future__ import annotations

from typing import Any

from src.control.agents.state import InterviewState
from src.data.repositories import (
    assessment_context_repository,
    interview_session_repository,
)
from src.utils.interview_graph import clean_section_name, utc_now_iso


def _max_turn_number(transcript: list[dict[str, Any]]) -> int:
    turns = [int(turn.get("turn_number") or 0) for turn in transcript]
    return max(turns or [0])


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _plan_sections(plan: dict[str, Any]) -> list[dict[str, Any]]:
    sections = plan.get("sections") if isinstance(plan, dict) else None
    if not isinstance(sections, list) or not sections:
        raise RuntimeError("Interview plan must include sections")

    normalized: list[dict[str, Any]] = []
    used_names: dict[str, int] = {}
    for index, item in enumerate(sections):
        if not isinstance(item, dict):
            continue
        section_name = str(
            item.get("section_name") or item.get("name") or f"section_{index + 1}"
        )
        clean_name = clean_section_name(section_name)
        count = used_names.get(clean_name, 0)
        used_names[clean_name] = count + 1
        if count:
            clean_name = f"{clean_name}_{count + 1}"
        normalized.append(
            {
                **item,
                "section_name": clean_name,
                "display_name": item.get("display_name") or section_name,
                "skill": item.get("skill"),
                "allocated_mins": float(item.get("allocated_mins") or 0),
                "expected_signals": item.get("expected_signals") or [],
            }
        )

    if not normalized:
        raise RuntimeError("Interview plan sections could not be normalized")
    return normalized


def _section_budgets(
    sections: list[dict[str, Any]],
    total_duration_secs: int,
) -> dict[str, int]:
    raw = {
        str(section["section_name"]): max(
            1,
            int(round(float(section.get("allocated_mins") or 0) * 60)),
        )
        for section in sections
    }
    allocated = sum(raw.values())
    if allocated <= 0:
        per_section = max(1, total_duration_secs // len(raw))
        return dict.fromkeys(raw, per_section)

    scale = total_duration_secs / allocated
    budgets = {key: max(1, int(round(value * scale))) for key, value in raw.items()}
    while sum(budgets.values()) < total_duration_secs:
        budgets[max(budgets, key=lambda key: budgets[key])] += 1
    while sum(budgets.values()) > total_duration_secs:
        target = max((key for key in budgets if budgets[key] > 1), default=None)
        if target is None:
            break
        budgets[target] -= 1
    return budgets


async def init_or_resume_session(state: InterviewState) -> dict[str, Any]:
    candidate_assessment_id = str(state["candidate_assessment_id"])
    session = await interview_session_repository.get_or_create_session(
        candidate_assessment_id
    )
    session_id = str(session["id"])
    transcript = list(session.get("transcript") or [])

    await interview_session_repository.mark_session_in_progress(session_id)
    await assessment_context_repository.mark_candidate_started(candidate_assessment_id)

    context = await assessment_context_repository.load_interview_context(
        candidate_assessment_id
    )
    if not context:
        raise RuntimeError(
            f"Candidate assessment context not found: {candidate_assessment_id}"
        )

    interview_plan = _dict(context.get("interview_plan"))
    duration_mins = int(
        interview_plan.get("total_mins") or context.get("interview_duration_mins") or 30
    )
    total_duration_secs = duration_mins * 60
    sections = _plan_sections(interview_plan)
    section_order = [str(section["section_name"]) for section in sections]
    current_section = clean_section_name(
        state.get("current_section") or section_order[0]
    )
    current_section_index = section_order.index(current_section)
    section_budgets = _section_budgets(sections, total_duration_secs)
    current_budget = int(section_budgets.get(current_section, total_duration_secs))
    elapsed_secs = int(session.get("total_elapsed_secs") or 0)
    current_section_elapsed = int(state.get("current_section_elapsed_secs") or 0)
    current = sections[current_section_index]
    resume_parsed = _dict(context.get("resume_parsed"))

    return {
        "candidate_assessment_id": candidate_assessment_id,
        "thread_id": candidate_assessment_id,
        "interview_session_id": session_id,
        "candidate_name": str(context.get("candidate_name") or "Candidate"),
        "candidate_email": context.get("candidate_email"),
        "role_name": str(context.get("role_name") or "the role"),
        "company_name": str(context.get("company_name") or ""),
        "assessment_title": str(context.get("assessment_title") or "Interview"),
        "resume_parsed": resume_parsed,
        "focus_areas": context.get("focus_areas"),
        "interview_plan": interview_plan,
        "runtime_sections": sections,
        "interview_duration_mins": duration_mins,
        "total_duration_secs": total_duration_secs,
        "remaining_secs": max(0, total_duration_secs - elapsed_secs),
        "soft_total_overrun_secs": max(0, elapsed_secs - total_duration_secs),
        "section_order": section_order,
        "section_budgets": section_budgets,
        "section_started_at": state.get("section_started_at") or utc_now_iso(),
        "current_section": current_section,
        "current_section_index": current_section_index,
        "current_section_name": current_section,
        "current_section_elapsed_secs": current_section_elapsed,
        "current_section_remaining_secs": max(
            0,
            current_budget - current_section_elapsed,
        ),
        "soft_section_overrun_secs": max(
            0,
            current_section_elapsed - current_budget,
        ),
        "force_close_due_to_overrun": False,
        "force_transition_due_to_overrun": False,
        "force_behavioural_cultural_due_to_time": False,
        "close_after_behavioural_cultural": False,
        "current_skill": current.get("skill"),
        "current_difficulty": state.get("current_difficulty") or "medium",
        "session_status": "IN_PROGRESS",
        "turn_number": _max_turn_number(transcript),
        "recent_turns": transcript[-8:],
        "elapsed_secs": elapsed_secs,
        "total_pause_secs": int(session.get("total_pause_secs") or 0),
        "asked_questions": list(state.get("asked_questions") or []),
        "skill_progress": dict(state.get("skill_progress") or {}),
        "live_evaluations": list(state.get("live_evaluations") or []),
        "latest_evaluation": state.get("latest_evaluation"),
        "clarification_count_for_current_question": int(
            state.get("clarification_count_for_current_question") or 0
        ),
        "silence_count_for_current_question": int(
            state.get("silence_count_for_current_question") or 0
        ),
        "non_answer_count_for_current_question": int(
            state.get("non_answer_count_for_current_question") or 0
        ),
        "skip_count_for_current_question": int(
            state.get("skip_count_for_current_question") or 0
        ),
        "think_silence_count": int(state.get("think_silence_count") or 0),
        "awaiting_think_confirmation": bool(
            state.get("awaiting_think_confirmation") or False
        ),
        "think_extension_active": bool(state.get("think_extension_active") or False),
        "irrelevant_count_total": int(state.get("irrelevant_count_total") or 0),
        "started_at": state.get("started_at") or utc_now_iso(),
        "grace_period_expires_at": None,
        "reconnect_count": int(state.get("reconnect_count") or 0),
        "resumed": bool(transcript),
        "closing_done": False,
        "final_evaluation_status": None,
        "holistic_evaluation_task_id": None,
        "should_close": False,
        "next_action": None,
        "next_node": "generate_opening_message",
    }
