"""
init_session_node — Initialises the interview session.

Runs once when the candidate clicks Start Interview. Loads context,
computes section time budgets, and writes the interview_sessions row.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)


async def init_session_node(state: InterviewState) -> dict:
    """
    Bootstrap the interview session.

    - Loads interview plan, JD analysis, and resume context into state.
    - Computes per-section time budgets.
    - Starts the session timer.
    """
    assessment_id = state.get("candidate_assessment_id", "unknown")

    logger.info("Initialising interview session for assessment=%s", assessment_id)

    # Load actual candidate resume, JD analysis, and interview plan from database
    interview_plan = None
    jd_analysis = None
    resume_context = None
    sections = None

    if assessment_id != "unknown":
        import json
        import uuid
        from typing import Any

        from sqlalchemy import text

        from src.data.clients.postgres_client import get_session_factory

        def parse_db_json(val: Any) -> Any:
            if not val:
                return None
            if isinstance(val, dict | list):
                return val
            if isinstance(val, str):
                try:
                    return json.loads(val)
                except Exception:
                    logger.exception("Failed to parse JSON string from DB: %s", val)
                    return None
            return None

        try:
            ca_uuid = uuid.UUID(str(assessment_id))
            SessionLocal = await get_session_factory()
            async with SessionLocal() as db_session:
                query = text(
                    "SELECT ca.resume_parsed, a.jd_analysis, a.interview_plan, a.role_name "
                    "FROM candidate_assessments ca "
                    "JOIN assessments a ON ca.assessment_id = a.id "
                    "WHERE ca.id = :ca_id"
                )
                result = await db_session.execute(query, {"ca_id": ca_uuid})
                row = result.fetchone()
                if row:
                    db_resume_parsed, db_jd_analysis, db_interview_plan, role_name = row

                    parsed_plan = parse_db_json(db_interview_plan)
                    parsed_jd = parse_db_json(db_jd_analysis)
                    parsed_resume = parse_db_json(db_resume_parsed)

                    if parsed_plan:
                        interview_plan = dict(parsed_plan)
                        role_title = role_name
                        if not role_title and parsed_jd:
                            role_title = parsed_jd.get("inferred_role_title")
                        interview_plan["role_title"] = role_title

                        # Populate sections based on the actual interview plan
                        if "sections" in parsed_plan:
                            sections = []
                            for idx, sec in enumerate(parsed_plan["sections"]):
                                sections.append(
                                    {
                                        "name": sec.get(
                                            "section_name", f"Section {idx + 1}"
                                        ),
                                        "skill": sec.get("skill") or "general",
                                        "priority_score": int(sec.get("priority_score"))
                                        if sec.get("priority_score") is not None
                                        else 5,
                                        "time_budget_secs": int(
                                            sec.get("allocated_mins", 5) * 60
                                        ),
                                        "time_elapsed_secs": 0,
                                        "questions_asked": 0,
                                        "concepts_covered": [],
                                        "is_complete": False,
                                    }
                                )

                    if parsed_jd:
                        jd_analysis = {
                            "role": parsed_jd.get(
                                "inferred_role_title", role_name or "Software Engineer"
                            ),
                            "key_skills": [
                                s.get("skill")
                                for s in parsed_jd.get("skills", [])
                                if s.get("skill")
                            ]
                            if "skills" in parsed_jd
                            else [],
                            "priority_signals": parsed_jd.get(
                                "behavioural_signals", []
                            ),
                        }

                    if parsed_resume:
                        summary = (
                            parsed_resume.get("summary")
                            or parsed_resume.get("Summary")
                            or "Not available"
                        )
                        skills = (
                            parsed_resume.get("skills")
                            or parsed_resume.get("Skills")
                            or []
                        )
                        experience_years = (
                            parsed_resume.get("experience_years")
                            or parsed_resume.get("ExperienceYears")
                            or parsed_resume.get("experience")
                            or 0
                        )

                        try:
                            exp_years = float(experience_years)
                        except (ValueError, TypeError):
                            exp_years = 0.0

                        resume_context = {
                            "summary": summary,
                            "skills": skills
                            if isinstance(skills, list)
                            else [skills]
                            if skills
                            else [],
                            "experience_years": exp_years,
                        }
                else:
                    logger.warning(
                        "No candidate assessment found for ID: %s", assessment_id
                    )
        except Exception:
            logger.exception(
                "Failed to load candidate assessment context from database"
            )

    # Compute initial section time remaining from first section
    first_section_time = sections[0]["time_budget_secs"] if sections else 300

    now_iso = datetime.now(UTC).isoformat()

    return {
        "interview_plan": interview_plan,
        "jd_analysis": jd_analysis,
        "resume_context": resume_context,
        "sections": sections,
        "current_section_index": 0,
        "current_section_time_remaining_secs": first_section_time,
        "turn_number": 0,
        "current_question_text": "",
        "current_question_difficulty": "easy",
        "consecutive_strong": 0,
        "consecutive_weak": 0,
        "used_concepts": [],
        "irrelevant_strike_count": 0,
        "silence_attempt": 0,
        "think_timer_active": False,
        "last_transcript": "",
        "last_response_classification": "",
        "last_evaluation": None,
        "transcript_turns": [],
        "answer_evaluations": [],
        "bot_reply_text": "",
        "bot_reply_type": "",
        "session_status": "in_progress",
        "timer_started_at": now_iso,
        "total_elapsed_secs": 0,
        "total_pause_secs": 0,
        "paused_at": None,
        "grace_period_expires_at": None,
        "auto_submit_triggered": False,
    }
