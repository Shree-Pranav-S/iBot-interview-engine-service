"""Load all prerequisite interview context before any bot speech."""

from __future__ import annotations

import secrets
from typing import Any

from src.clients.core_api_client import get_core_api_client
from src.control.agents.state import InterviewState
from src.control.agents.utils.initialize_context import (
    _experience_years,
    _runtime_sections,
    _skills,
)
from src.core.exceptions import (
    InterviewContextNotFoundException,
    InterviewPlanMissingException,
)


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
        raise InterviewContextNotFoundException(
            f"Candidate assessment context not found: {candidate_assessment_id}"
        )

    interview_plan = context.get("interview_plan")
    if not isinstance(interview_plan, dict):
        raise InterviewPlanMissingException(
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
    jd_analysis = context.get("jd_analysis")
    runtime_sections = _runtime_sections(
        interview_plan,
        role_name=str(context.get("role_name") or ""),
        jd_analysis=jd_analysis if isinstance(jd_analysis, dict) else None,
    )
    intro_index = next(
        (
            index
            for index, section in enumerate(runtime_sections)
            if section["section_kind"] == "self_intro"
        ),
        0,
    )
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
        "current_section_index": intro_index,
        "current_section_kind": "self_intro",
        "current_technical_skill": None,
        "current_question_id": "phase-1:self-introduction",
        "current_question_text": (
            "Could you tell me about your professional background?"
        ),
        "last_rephrased_question": None,
        "current_question_difficulty": None,
        "candidate_event": None,
        "previous_candidate_response": "",
        "llm_key_slot": None,
        "last_classification": None,
        "classification_source": None,
        "last_response_type": None,
        "last_response_substantial": None,
        "latest_evaluation": None,
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
        "self_intro_accumulated_response": "",
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
        "section_budgets_secs": section_budgets_secs,
        "current_section_budget_secs": current_section_budget_secs,
        "current_section_started_elapsed_secs": elapsed_secs,
        "current_section_elapsed_secs": 0,
        "current_section_remaining_secs": max(
            0,
            intro_deadline - elapsed_secs,
        ),
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
