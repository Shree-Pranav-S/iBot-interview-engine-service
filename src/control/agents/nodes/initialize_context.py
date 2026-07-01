"""Load all prerequisite interview context before any bot speech."""

from __future__ import annotations

import secrets
from typing import Any

from src.clients.core_api_client import get_core_api_client
from src.control.agents.state import InterviewState


def _skills(value: Any) -> list[str]:
    """
    Extract a unique list of candidate skills from the resume payload.

    Args:
        value: A list of string skills or dict objects containing skill names.

    Returns:
        A normalized, deduplicated list of strings.
    """
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str):
            skill = item.strip()
        elif isinstance(item, dict):
            skill = str(
                item.get("skill") or item.get("name") or item.get("title") or ""
            ).strip()
        else:
            skill = ""

        key = skill.casefold()
        if skill and key not in seen:
            normalized.append(skill)
            seen.add(key)
    return normalized


def _experience_years(value: Any) -> float:
    """
    Safely extract candidate experience years from the resume payload.

    Args:
        value: A raw number or string representing years of experience.

    Returns:
        A float representing years, defaulting to 0.0 on error.
    """
    try:
        return max(0.0, float(value or 0))
    except (TypeError, ValueError):
        return 0.0


def _runtime_sections(interview_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Normalize the stored interview plan into an actionable sequence of runtime sections.
    This ensures each section has an explicitly defined kind (technical, behavioural)
    and validates time allocations.

    Args:
        interview_plan: The stored plan dictionary for the candidate.

    Returns:
        A standardized list of section dictionaries.

    Raises:
        RuntimeError: If the plan is malformed or missing sections.
    """
    raw_sections = interview_plan.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise RuntimeError("Interview plan must contain at least one section")

    sections: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_sections):
        if not isinstance(raw, dict):
            continue
        section_name = str(raw.get("section_name") or "").strip()
        skill = str(raw.get("skill") or "").strip() or None
        if section_name == "self_intro":
            kind = "self_intro"
        elif section_name == "behavioural_cultural":
            kind = "behavioural_cultural"
        else:
            kind = "technical"
            if not skill:
                skill = section_name

        expected = raw.get("expected_signals")
        expected_signals = [
            str(item).strip()
            for item in (expected if isinstance(expected, list) else [])
            if str(item).strip()
        ]
        sections.append(
            {
                "index": index,
                "section_name": section_name or f"section_{index + 1}",
                "section_kind": kind,
                "skill": skill,
                "expected_signals": expected_signals,
                "allocated_mins": float(raw.get("allocated_mins") or 0),
            }
        )

    if not sections:
        raise RuntimeError("Interview plan sections could not be normalized")
    return sections


async def initialize_interview_context(state: InterviewState) -> dict[str, Any]:
    """
    Initialize DB session, company, plan, and the bounded resume context.

    This is the entry point node for any brand new interview. It fetches the
    candidate context from PostgreSQL, constructs the initial state dictionary,
    and sets up the time tracking structure.

    Args:
        state: The LangGraph input state (contains at minimum the candidate assessment ID).

    Returns:
        The fully initialized InterviewState dictionary.
    """

    candidate_assessment_id = str(state["candidate_assessment_id"])
    context, session = await get_core_api_client().initialize_interview_context(
        candidate_assessment_id,
    )
    if not context:
        raise RuntimeError(
            f"Candidate assessment context not found: {candidate_assessment_id}"
        )

    interview_plan = context.get("interview_plan")
    if not isinstance(interview_plan, dict):
        raise RuntimeError(
            "Interview plan is missing for candidate assessment: "
            f"{candidate_assessment_id}"
        )

    session_id = str(session["id"])

    resume_parsed = context.get("resume_parsed")
    resume = resume_parsed if isinstance(resume_parsed, dict) else {}
    resume_context = {
        "skills": _skills(resume.get("skills")),
        "experience_years": _experience_years(resume.get("experience_years")),
    }
    runtime_sections = _runtime_sections(interview_plan)
    intro_index = next(
        (
            index
            for index, section in enumerate(runtime_sections)
            if section["section_kind"] == "self_intro"
        ),
        0,
    )
    intro_section = runtime_sections[intro_index]
    duration_mins = int(
        interview_plan.get("total_mins") or context.get("interview_duration_mins") or 30
    )
    total_duration_secs = max(60, duration_mins * 60)
    section_budgets_secs = {
        str(index): max(
            1,
            int(round(float(section.get("allocated_mins") or 0) * 60)),
        )
        for index, section in enumerate(runtime_sections)
    }
    current_section_budget_secs = section_budgets_secs.get(
        str(intro_index),
        60,
    )
    elapsed_secs = int(session.get("total_elapsed_secs") or 0)
    reserved_future_section_secs = sum(
        section_budgets_secs.get(str(index), 0)
        for index, section in enumerate(runtime_sections)
        if index > intro_index
        and section.get("section_kind")
        in {
            "technical",
            "behavioural_cultural",
        }
    )
    intro_deadline = min(
        elapsed_secs + current_section_budget_secs,
        max(0, total_duration_secs - reserved_future_section_secs),
    )

    return {
        "candidate_assessment_id": candidate_assessment_id,
        "interview_session_id": session_id,
        "candidate_name": str(context.get("candidate_name") or "Candidate"),
        "company_name": str(context.get("company_name") or "the company"),
        # Deliberately do not retain the complete resume_parsed payload.
        "resume_context": resume_context,
        "inferred_difficulty": str(
            interview_plan.get("inferred_difficulty") or "mid-level"
        )
        .strip()
        .lower(),
        "runtime_sections": runtime_sections,
        "current_section": str(intro_section["section_name"]),
        "current_section_index": intro_index,
        "current_section_kind": "self_intro",
        "current_technical_skill": None,
        "current_expected_signals": [],
        "current_question_id": "phase-1:self-introduction",
        "current_question_text": (
            "Could you tell me about your professional background?"
        ),
        "last_rephrased_question": None,
        "last_spoken_opening_text": None,
        "current_question_difficulty": None,
        "is_self_introduction": True,
        "candidate_event": None,
        "previous_candidate_response": "",
        "previous_response_duration_ms": None,
        "llm_key_slot": None,
        "last_classification": None,
        "classification_source": None,
        "last_response_type": None,
        "last_response_substantial": None,
        "latest_evaluation": None,
        "evaluation_source": None,
        "asked_questions": [],
        "used_topics_by_skill": {},
        "skill_evaluation_streaks": {},
        "question_variation_seed": secrets.token_hex(8),
        "probe_deeper": False,
        "thread_follow_up_used": False,
        "last_skip_resume_skill_match": False,
        "bot_reply_text": "",
        "bot_reply_type": "",
        "response_preface_text": None,
        "silence_stage": "none",
        "skip_attempts_for_current_question": 0,
        "self_intro_elaboration_requested": False,
        "should_advance_question": False,
        "phase_complete": False,
        "next_action": "deliver_opening",
        "turn_number": 1,
        "pending_bot_turn": None,
        "pending_candidate_turn": None,
        "violations_to_persist": [],
        "recent_violations": list(session.get("violations") or [])[-20:],
        "total_duration_secs": total_duration_secs,
        "elapsed_secs": elapsed_secs,
        "remaining_secs": max(0, total_duration_secs - elapsed_secs),
        "section_budgets_secs": section_budgets_secs,
        "current_section_budget_secs": current_section_budget_secs,
        "current_section_started_elapsed_secs": elapsed_secs,
        "current_section_elapsed_secs": 0,
        "current_section_remaining_secs": max(
            0,
            intro_deadline - elapsed_secs,
        ),
        "reserved_future_section_secs": reserved_future_section_secs,
        "current_section_transition_deadline_elapsed_secs": intro_deadline,
        "pending_section_index": None,
        "suppress_previous_context_for_next_question": False,
        "transition_reason": None,
        "barge_in_triggered": False,
        "self_intro_leftover_redistributed": False,
        # This stays false until LiveKit reports that bot audio started.
        "timer_started": False,
        "timer_started_at": None,
        "session_status": "IN_PROGRESS",
        "should_close": False,
        "closing_done": False,
        "closing_reason": None,
        "holistic_evaluation_status": None,
        "holistic_evaluation_task_id": None,
    }
