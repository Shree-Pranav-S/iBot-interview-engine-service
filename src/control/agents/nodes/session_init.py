"""Session initialization node."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from src.control.agents.state import InterviewState, SectionState
from src.control.session_loader import normalize_resume, normalize_sections
from src.data.repositories import interview_workflow_repository as db


def _last_question_from_transcript(transcript: list[dict]) -> str:
    for turn in reversed(transcript):
        if turn.get("speaker") == "bot" and turn.get("turn_type") == "question":
            return str(turn.get("text") or "")
    return ""


def _section_index_from_transcript(
    transcript: list[dict], sections: list[SectionState]
) -> int:
    for turn in reversed(transcript):
        if turn.get("section"):
            section_name = str(turn.get("section"))
            for idx, section in enumerate(sections):
                if (
                    str(section.get("section_name") or section.get("name"))
                    == section_name
                ):
                    return idx
    return 0


def _build_sections(
    interview_plan: dict | None, duration_mins: int | None
) -> list[SectionState]:
    normalized = normalize_sections(interview_plan, duration_mins)
    sections: list[SectionState] = []
    for section in normalized:
        name = str(section.get("section_name") or section.get("name") or "general")
        concepts = list(section.get("concepts_covered") or [])
        sections.append(
            {
                **section,
                "name": name,
                "section_name": name,
                "skill": section.get("skill"),
                "priority_score": section.get("priority_score"),
                "allocated_mins": float(section.get("allocated_mins") or 1.0),
                "time_budget_secs": int(section.get("time_budget_secs") or 60),
                "time_elapsed_secs": int(section.get("time_elapsed_secs") or 0),
                "questions_asked": int(section.get("questions_asked") or 0),
                "concepts_covered": concepts,
                "is_complete": bool(section.get("is_complete") or False),
            }
        )
    return sections


async def session_init(state: InterviewState) -> dict:
    ca_id = state["candidate_assessment_id"]
    context = await db.load_candidate_context(ca_id)
    if not context:
        raise ValueError(f"Candidate assessment {ca_id} not found.")

    session = await db.create_or_resume_session(ca_id)
    sections = _build_sections(
        context.get("interview_plan"),
        context.get("interview_duration_mins"),
    )
    now = time.time()
    existing_transcript = session.get("transcript") or []
    existing_violations = session.get("violations") or []
    resumed = bool(existing_transcript)
    current_idx = (
        _section_index_from_transcript(existing_transcript, sections) if resumed else 0
    )
    if current_idx >= len(sections):
        current_idx = max(0, len(sections) - 1)
    current_section = sections[current_idx] if sections else {}
    current_question = _last_question_from_transcript(existing_transcript)
    allocated_secs = float(current_section.get("time_budget_secs") or 60)
    elapsed_secs = int(session.get("total_elapsed_secs") or 0)
    last_turn_number = max(
        [int(turn.get("turn_number") or 0) for turn in existing_transcript] or [0]
    )
    resume = normalize_resume(context.get("resume_parsed") or {})
    total_secs = sum(int(section.get("time_budget_secs") or 0) for section in sections)

    patch: dict = {
        "candidate_assessment_id": str(context["candidate_assessment_id"]),
        "role_name": context.get("role_name") or "the role",
        "company_name": context.get("company_name") or "the company",
        "interview_plan": context.get("interview_plan") or {},
        "resume_parsed": resume,
        "resume_context": resume,
        "sections": sections,
        "session_id": str(session["id"]),
        "session_status": "IN_PROGRESS" if resumed else "INITIALIZING",
        "timer_started_at": datetime.now(UTC).isoformat(),
        "interview_started_at": now - elapsed_secs,
        "total_elapsed_secs": elapsed_secs,
        "total_pause_secs": int(session.get("total_pause_secs") or 0),
        "paused_at": None,
        "grace_period_expires_at": None,
        "auto_submit_triggered": False,
        "total_interview_allocated_secs": total_secs,
        "current_section_index": current_idx,
        "current_section_name": str(current_section.get("section_name") or "general"),
        "current_section_time_remaining_secs": int(allocated_secs),
        "section_started_at": now,
        "section_allocated_secs": allocated_secs,
        "questions_asked_in_section": 1 if current_question else 0,
        "concepts_covered_in_section": list(
            current_section.get("concepts_covered") or []
        ),
        "used_concepts": list(current_section.get("concepts_covered") or []),
        "current_question": current_question,
        "current_question_text": current_question,
        "current_question_difficulty": "easy",
        "current_question_concept": "",
        "turn_number": last_turn_number,
        "last_bot_text": "",
        "last_transcript": "",
        "last_response_classification": None,
        "candidate_raw_text": "",
        "candidate_stt_confidence": None,
        "response_class": None,
        "last_evaluation": None,
        "bot_reply_text": "",
        "bot_reply_type": "",
        "irrelevant_count": len(
            [
                item
                for item in existing_violations
                if item.get("violation_type") == "irrelevant"
            ]
        ),
        "irrelevant_strike_count": len(
            [
                item
                for item in existing_violations
                if item.get("violation_type") == "irrelevant"
            ]
        ),
        "awaiting_think_decision": False,
        "think_timer_active": False,
        "skip_requested": False,
        "current_difficulty_level": 1,
        "next_question_mode": "normal",
        "last_question_was_weak_retry": False,
        "should_close": False,
        "closing_done": False,
        "holistic_evaluation_done": False,
        "next_node": "await_response" if resumed else "opening",
        "resumed": resumed,
        "violations": existing_violations or [],
        "question_scores": [],
    }

    if resumed and current_question:
        text = f"Welcome back. Let me repeat the current question: {current_question}"
        turn_number = last_turn_number + 1
        patch.update(
            {
                "last_bot_text": text,
                "bot_reply_text": text,
                "bot_reply_type": "question",
                "turn_number": turn_number,
                "transcript": [
                    *existing_transcript,
                    {
                        "turn_number": turn_number,
                        "speaker": "bot",
                        "text": text,
                        "tone": "neutral",
                        "section": patch["current_section_name"],
                        "turn_type": "question",
                        "difficulty": patch["current_question_difficulty"],
                    },
                ],
            }
        )
    else:
        patch["transcript"] = []

    await db.mark_session_in_progress(ca_id)
    return patch
