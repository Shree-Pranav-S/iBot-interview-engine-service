"""Raw SQL helpers for interview workflow persistence."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from src.data.clients.postgres_client import get_session_factory

logger = logging.getLogger(__name__)


def _uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


async def fetch_one(query: str, params: dict[str, Any]) -> dict:
    SessionLocal = await get_session_factory()
    async with SessionLocal() as session:
        result = await session.execute(text(query), params)
        row = result.mappings().first()
        return dict(row) if row else {}


async def fetch_one_write(query: str, params: dict[str, Any]) -> dict:
    SessionLocal = await get_session_factory()
    async with SessionLocal() as session:
        result = await session.execute(text(query), params)
        row = result.mappings().first()
        await session.commit()
        return dict(row) if row else {}


async def execute(query: str, params: dict[str, Any]) -> None:
    SessionLocal = await get_session_factory()
    async with SessionLocal() as session:
        await session.execute(text(query), params)
        await session.commit()


async def load_candidate_context(candidate_assessment_id: str) -> dict:
    return await fetch_one(
        """
        SELECT
            ca.id AS candidate_assessment_id,
            ca.status AS candidate_assessment_status,
            ca.resume_parsed,
            a.title AS assessment_title,
            a.role_name,
            a.interview_plan,
            a.interview_duration_mins,
            COALESCE(r.company_name, '') AS company_name
        FROM candidate_assessments ca
        JOIN assessments a ON ca.assessment_id = a.id
        LEFT JOIN recruiters r ON a.recruiter_id = r.id
        WHERE ca.id = :candidate_assessment_id
        """,
        {"candidate_assessment_id": _uuid(candidate_assessment_id)},
    )


async def get_existing_session(candidate_assessment_id: str) -> dict:
    return await fetch_one(
        """
        SELECT id, status, transcript, violations, total_elapsed_secs,
               total_pause_secs, grace_period_expires_at
        FROM interview_sessions
        WHERE candidate_assessment_id = :candidate_assessment_id
        """,
        {"candidate_assessment_id": _uuid(candidate_assessment_id)},
    )


async def assert_session_can_start(candidate_assessment_id: str) -> tuple[bool, str]:
    context = await load_candidate_context(candidate_assessment_id)
    if not context:
        return False, "Candidate assessment was not found."

    ca_status = str(context.get("candidate_assessment_status") or "")
    if ca_status in {"COMPLETED", "EVALUATED"}:
        return False, "This interview has already been completed."

    session = await get_existing_session(candidate_assessment_id)
    if not session:
        return True, ""

    session_status = str(session.get("status") or "")
    if session_status in {"COMPLETED", "EVALUATED", "TERMINATED", "DEACTIVATED"}:
        return False, "This interview session is already closed."

    expires_at = session.get("grace_period_expires_at")
    if session_status == "PAUSED" and expires_at is not None:
        now = datetime.now(UTC)
        if expires_at < now:
            await mark_session_deactivated(candidate_assessment_id)
            return False, "The reconnection grace period has expired."

    return True, ""


async def create_or_resume_session(candidate_assessment_id: str) -> dict:
    existing = await get_existing_session(candidate_assessment_id)
    if existing:
        return existing

    row = await fetch_one_write(
        """
        INSERT INTO interview_sessions (
            candidate_assessment_id,
            transcript,
            violations,
            total_elapsed_secs,
            total_pause_secs,
            status
        )
        VALUES (
            :candidate_assessment_id,
            '[]'::jsonb,
            '[]'::jsonb,
            0,
            0,
            'INITIALIZING'
        )
        RETURNING id, status, transcript, violations, total_elapsed_secs,
                  total_pause_secs, grace_period_expires_at
        """,
        {"candidate_assessment_id": _uuid(candidate_assessment_id)},
    )
    return row


async def mark_candidate_started(candidate_assessment_id: str) -> None:
    await execute(
        """
        UPDATE candidate_assessments
        SET status = 'IN_PROGRESS',
            interview_started_at = COALESCE(interview_started_at, NOW()),
            updated_at = NOW()
        WHERE id = :candidate_assessment_id
        """,
        {"candidate_assessment_id": _uuid(candidate_assessment_id)},
    )


async def mark_candidate_finished(candidate_assessment_id: str) -> None:
    await execute(
        """
        UPDATE candidate_assessments
        SET status = 'COMPLETED',
            interview_ended_at = COALESCE(interview_ended_at, NOW()),
            updated_at = NOW()
        WHERE id = :candidate_assessment_id
        """,
        {"candidate_assessment_id": _uuid(candidate_assessment_id)},
    )


async def mark_session_in_progress(candidate_assessment_id: str) -> None:
    session = await get_existing_session(candidate_assessment_id)
    add_pause = 0
    expires_at = session.get("grace_period_expires_at") if session else None
    if str(session.get("status") if session else "") == "PAUSED" and expires_at:
        pause_started_at = expires_at - timedelta(minutes=5)
        add_pause = max(0, int((datetime.now(UTC) - pause_started_at).total_seconds()))

    await execute(
        """
        UPDATE interview_sessions
        SET status = 'IN_PROGRESS',
            total_pause_secs = total_pause_secs + :add_pause,
            grace_period_expires_at = NULL,
            last_updated_at = NOW()
        WHERE candidate_assessment_id = :candidate_assessment_id
        """,
        {
            "candidate_assessment_id": _uuid(candidate_assessment_id),
            "add_pause": add_pause,
        },
    )


async def mark_session_paused(candidate_assessment_id: str) -> None:
    await execute(
        """
        UPDATE interview_sessions
        SET status = 'PAUSED',
            grace_period_expires_at = NOW() + INTERVAL '5 minutes',
            last_updated_at = NOW()
        WHERE candidate_assessment_id = :candidate_assessment_id
          AND status = 'IN_PROGRESS'
        """,
        {"candidate_assessment_id": _uuid(candidate_assessment_id)},
    )


async def mark_session_deactivated(candidate_assessment_id: str) -> None:
    await execute(
        """
        UPDATE interview_sessions
        SET status = 'DEACTIVATED',
            last_updated_at = NOW()
        WHERE candidate_assessment_id = :candidate_assessment_id
        """,
        {"candidate_assessment_id": _uuid(candidate_assessment_id)},
    )


async def persist_session_state(state: dict) -> None:
    if not state.get("session_id"):
        return
    await execute(
        """
        UPDATE interview_sessions
        SET transcript = CAST(:transcript AS jsonb),
            violations = CAST(:violations AS jsonb),
            total_elapsed_secs = :total_elapsed_secs,
            total_pause_secs = :total_pause_secs,
            status = :status,
            last_updated_at = NOW()
        WHERE id = :session_id
        """,
        {
            "session_id": _uuid(state["session_id"]),
            "transcript": json.dumps(state.get("transcript") or []),
            "violations": json.dumps(state.get("violations") or []),
            "total_elapsed_secs": int(state.get("total_elapsed_secs") or 0),
            "total_pause_secs": int(state.get("total_pause_secs") or 0),
            "status": state.get("session_status") or "IN_PROGRESS",
        },
    )


def _score_average(scores: list[dict]) -> float:
    if not scores:
        return 0.0
    return round(
        sum(float(item.get("raw_score") or 0.0) for item in scores) / len(scores), 2
    )


def build_evaluation_payload(state: dict) -> dict:
    scores = list(state.get("question_scores") or [])
    sections = list(state.get("sections") or [])
    section_priorities = {
        str(section.get("section_name")): section.get("priority_score")
        for section in sections
    }

    grouped: dict[str, list[dict]] = {}
    for score in scores:
        grouped.setdefault(str(score.get("section") or "general"), []).append(score)

    skill_scores: dict[str, dict] = {}
    section_summaries: dict[str, dict] = {}
    for section_name, section_scores in grouped.items():
        avg = _score_average(section_scores)
        present = sorted(
            {
                signal
                for score in section_scores
                for signal in score.get("signals_demonstrated", [])
            }
        )
        missing = sorted(
            {
                signal
                for score in section_scores
                for signal in score.get("signals_missing", [])
            }
        )
        priority = section_priorities.get(section_name) or 5.0
        skill_scores[section_name] = {
            "priority_score": priority,
            "raw_score": avg,
            "weighted_score": round(avg * (float(priority) / 10.0), 2),
            "difficulty_reached": max(
                (score.get("difficulty") == "hard") * 3
                or (score.get("difficulty") == "medium") * 2
                or 1
                for score in section_scores
            ),
            "signals_demonstrated": present[:8],
            "signals_missing": missing[:8],
            "summary": f"Average score {avg}/10 across {len(section_scores)} answer(s).",
        }
        section_summaries[section_name] = {
            "summary": f"Covered {len(section_scores)} question(s) in {section_name}.",
            "avg_score": avg,
            "questions_asked": len(section_scores),
        }

    technical_scores = [
        score
        for score in scores
        if str(score.get("section") or "").lower()
        not in {"self_intro", "behavioural", "behavioral", "cultural"}
    ]
    behavioural_scores = [
        score
        for score in scores
        if str(score.get("section") or "").lower() in {"behavioural", "behavioral"}
    ]
    cultural_scores = [
        score
        for score in scores
        if str(score.get("section") or "").lower() == "cultural"
    ]

    technical = _score_average(technical_scores) * 10
    behavioural = (
        _score_average(behavioural_scores) * 10 if behavioural_scores else 60.0
    )
    cultural = _score_average(cultural_scores) * 10 if cultural_scores else 60.0
    overall = round((technical * 0.6) + (behavioural * 0.2) + (cultural * 0.2), 2)

    best = max(
        scores, key=lambda score: float(score.get("raw_score") or 0.0), default=None
    )
    weakest = min(
        scores, key=lambda score: float(score.get("raw_score") or 0.0), default=None
    )
    violations = list(state.get("violations") or [])
    strengths = [
        signal for score in scores for signal in score.get("signals_demonstrated", [])
    ][:4]
    concerns = [
        signal for score in scores for signal in score.get("signals_missing", [])
    ][:4]

    if overall >= 85:
        recommendation = "STRONG_HIRE"
    elif overall >= 70:
        recommendation = "HIRE"
    elif overall >= 55:
        recommendation = "CONSIDER"
    elif overall >= 40:
        recommendation = "WEAK"
    else:
        recommendation = "NO_HIRE"

    return {
        "skill_scores": skill_scores
        or {
            "overall": {
                "raw_score": 0.0,
                "weighted_score": 0.0,
                "summary": "No scored answers.",
            }
        },
        "technical_dimension_score": technical,
        "score_evidence": strengths or ["Interview transcript captured for review."],
        "score_summary": f"Live interview aggregate score is {overall}/100.",
        "behavioural_score": behavioural,
        "behavioural_evidence": strengths
        or ["No separate behavioural evidence captured."],
        "behavioural_summary": "Behavioural score derived from live answer evaluations.",
        "cultural_fit_score": cultural,
        "cultural_fit_evidence": strengths
        or ["No separate cultural evidence captured."],
        "cultural_fit_summary": "Cultural fit score derived from live answer evaluations.",
        "section_summaries": section_summaries,
        "overall_score": overall,
        "hiring_recommendation": recommendation,
        "overall_narrative": (
            "This report was generated from the live AI interview transcript and "
            "turn-level scoring captured during the session."
        ),
        "strengths": strengths or ["Completed the interview session."],
        "concerns": concerns or ["No major concerns were identified by live scoring."],
        "violation_summary": {
            "total_irrelevant": sum(
                1 for item in violations if item.get("violation_type") == "irrelevant"
            ),
            "total_silences": sum(
                1 for item in violations if item.get("violation_type") == "silence"
            ),
            "terminated_early": state.get("session_status") == "TERMINATED",
            "entries": violations,
        }
        if violations
        else None,
        "best_answer": _highlight(best),
        "weakest_answer": _highlight(weakest),
        "recommendation_reasoning": "Recommendation is based on weighted live interview scores.",
    }


def _highlight(score: dict | None) -> dict | None:
    if not score:
        return None
    return {
        "question": score.get("question") or "",
        "turn_number": score.get("turn_number") or 0,
        "section": score.get("section"),
        "difficulty_at_time": score.get("difficulty"),
        "reason": score.get("reasoning") or "",
    }


async def upsert_interview_evaluation(state: dict) -> None:
    if not state.get("session_id"):
        return
    payload = build_evaluation_payload(state)
    await execute(
        """
        INSERT INTO interview_evaluations (
            candidate_assessment_id,
            session_id,
            skill_scores,
            technical_dimension_score,
            score_evidence,
            score_summary,
            behavioural_score,
            behavioural_evidence,
            behavioural_summary,
            cultural_fit_score,
            cultural_fit_evidence,
            cultural_fit_summary,
            tone_classification_score,
            tone_distribution,
            section_summaries,
            overall_score,
            hiring_recommendation,
            overall_narrative,
            strengths,
            concerns,
            violation_summary,
            best_answer,
            weakest_answer,
            recommendation_reasoning
        )
        VALUES (
            :candidate_assessment_id,
            :session_id,
            CAST(:skill_scores AS jsonb),
            :technical_dimension_score,
            :score_evidence,
            :score_summary,
            :behavioural_score,
            :behavioural_evidence,
            :behavioural_summary,
            :cultural_fit_score,
            :cultural_fit_evidence,
            :cultural_fit_summary,
            NULL,
            NULL,
            CAST(:section_summaries AS jsonb),
            :overall_score,
            :hiring_recommendation,
            :overall_narrative,
            :strengths,
            :concerns,
            CAST(:violation_summary AS jsonb),
            CAST(:best_answer AS jsonb),
            CAST(:weakest_answer AS jsonb),
            :recommendation_reasoning
        )
        ON CONFLICT (candidate_assessment_id) DO UPDATE SET
            skill_scores = EXCLUDED.skill_scores,
            technical_dimension_score = EXCLUDED.technical_dimension_score,
            score_evidence = EXCLUDED.score_evidence,
            score_summary = EXCLUDED.score_summary,
            behavioural_score = EXCLUDED.behavioural_score,
            behavioural_evidence = EXCLUDED.behavioural_evidence,
            behavioural_summary = EXCLUDED.behavioural_summary,
            cultural_fit_score = EXCLUDED.cultural_fit_score,
            cultural_fit_evidence = EXCLUDED.cultural_fit_evidence,
            cultural_fit_summary = EXCLUDED.cultural_fit_summary,
            section_summaries = EXCLUDED.section_summaries,
            overall_score = EXCLUDED.overall_score,
            hiring_recommendation = EXCLUDED.hiring_recommendation,
            overall_narrative = EXCLUDED.overall_narrative,
            strengths = EXCLUDED.strengths,
            concerns = EXCLUDED.concerns,
            violation_summary = EXCLUDED.violation_summary,
            best_answer = EXCLUDED.best_answer,
            weakest_answer = EXCLUDED.weakest_answer,
            recommendation_reasoning = EXCLUDED.recommendation_reasoning,
            generated_at = NOW()
        """,
        {
            "candidate_assessment_id": _uuid(state["candidate_assessment_id"]),
            "session_id": _uuid(state["session_id"]),
            "skill_scores": json.dumps(payload["skill_scores"]),
            "technical_dimension_score": payload["technical_dimension_score"],
            "score_evidence": payload["score_evidence"],
            "score_summary": payload["score_summary"],
            "behavioural_score": payload["behavioural_score"],
            "behavioural_evidence": payload["behavioural_evidence"],
            "behavioural_summary": payload["behavioural_summary"],
            "cultural_fit_score": payload["cultural_fit_score"],
            "cultural_fit_evidence": payload["cultural_fit_evidence"],
            "cultural_fit_summary": payload["cultural_fit_summary"],
            "section_summaries": json.dumps(payload["section_summaries"]),
            "overall_score": payload["overall_score"],
            "hiring_recommendation": payload["hiring_recommendation"],
            "overall_narrative": payload["overall_narrative"],
            "strengths": payload["strengths"],
            "concerns": payload["concerns"],
            "violation_summary": json.dumps(payload["violation_summary"]),
            "best_answer": json.dumps(payload["best_answer"]),
            "weakest_answer": json.dumps(payload["weakest_answer"]),
            "recommendation_reasoning": payload["recommendation_reasoning"],
        },
    )
