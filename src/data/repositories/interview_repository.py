import json
import logging
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import text

from src.data.clients.postgres_client import get_session_factory

logger = logging.getLogger(__name__)


def _parse_datetime(dt_str: str | None) -> datetime | None:
    if not dt_str:
        return None
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


async def sync_interview_state(state: dict) -> None:
    """
    Synchronizes the LangGraph InterviewState dictionary with the Postgres
    relational schema. Kept as a no-op because transcript and violations are
    persisted directly on interview_sessions in the current schema.
    """
    logger.debug("sync_interview_state skipped (no-op)")


async def get_interview_data_for_evaluation(
    candidate_assessment_id: uuid.UUID,
) -> dict[str, Any] | None:
    """Fetch the durable session JSON used for holistic assessment."""
    try:
        SessionLocal = await get_session_factory()
        async with SessionLocal() as db_session:
            result = await db_session.execute(
                text(
                    """
                    SELECT id, candidate_assessment_id, status, transcript, violations
                    FROM interview_sessions
                    WHERE candidate_assessment_id = :candidate_assessment_id
                    """
                ),
                {"candidate_assessment_id": candidate_assessment_id},
            )
            session = result.mappings().first()
            if not session:
                return None
            return {"session": dict(session), "turns": [], "evaluations": []}
    except Exception:
        logger.exception(
            "Failed to fetch interview data for %s", candidate_assessment_id
        )
        return None


def _as_text_array(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _json_dumps(value: Any) -> str:
    return json.dumps(value)


async def save_holistic_evaluation(
    candidate_assessment_id: uuid.UUID, session_id: uuid.UUID, eval_data: dict[str, Any]
) -> None:
    """Upsert the final holistic evaluation report to Postgres."""
    try:
        SessionLocal = await get_session_factory()
        async with SessionLocal() as db_session:
            await db_session.execute(
                text(
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
                        :tone_classification_score,
                        CAST(:tone_distribution AS jsonb),
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
                        session_id = EXCLUDED.session_id,
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
                        tone_classification_score = EXCLUDED.tone_classification_score,
                        tone_distribution = EXCLUDED.tone_distribution,
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
                    """
                ),
                {
                    "candidate_assessment_id": candidate_assessment_id,
                    "session_id": session_id,
                    "skill_scores": _json_dumps(eval_data.get("skill_scores") or {}),
                    "technical_dimension_score": float(
                        eval_data.get("technical_dimension_score") or 0.0
                    ),
                    "score_evidence": _as_text_array(eval_data.get("score_evidence")),
                    "score_summary": str(eval_data.get("score_summary") or ""),
                    "behavioural_score": float(
                        eval_data.get("behavioural_score") or 0.0
                    ),
                    "behavioural_evidence": _as_text_array(
                        eval_data.get("behavioural_evidence")
                    ),
                    "behavioural_summary": str(
                        eval_data.get("behavioural_summary") or ""
                    ),
                    "cultural_fit_score": float(
                        eval_data.get("cultural_fit_score") or 0.0
                    ),
                    "cultural_fit_evidence": _as_text_array(
                        eval_data.get("cultural_fit_evidence")
                    ),
                    "cultural_fit_summary": str(
                        eval_data.get("cultural_fit_summary") or ""
                    ),
                    "tone_classification_score": eval_data.get(
                        "tone_classification_score"
                    ),
                    "tone_distribution": _json_dumps(
                        eval_data.get("tone_distribution")
                    ),
                    "section_summaries": _json_dumps(
                        eval_data.get("section_summaries") or {}
                    ),
                    "overall_score": float(eval_data.get("overall_score") or 0.0),
                    "hiring_recommendation": str(
                        eval_data.get("hiring_recommendation") or "CONSIDER"
                    ),
                    "overall_narrative": str(eval_data.get("overall_narrative") or ""),
                    "strengths": _as_text_array(eval_data.get("strengths")),
                    "concerns": _as_text_array(eval_data.get("concerns")),
                    "violation_summary": _json_dumps(
                        eval_data.get("violation_summary")
                    ),
                    "best_answer": _json_dumps(eval_data.get("best_answer")),
                    "weakest_answer": _json_dumps(eval_data.get("weakest_answer")),
                    "recommendation_reasoning": str(
                        eval_data.get("recommendation_reasoning") or ""
                    ),
                },
            )
            await db_session.commit()
    except Exception:
        logger.exception(
            "Failed to save holistic evaluation for %s", candidate_assessment_id
        )
        raise


async def get_assessment_context_by_ca_id(ca_uuid: uuid.UUID) -> tuple | None:
    """Fetches resume, JD analysis, interview plan, and role name for an assessment."""
    try:
        SessionLocal = await get_session_factory()
        async with SessionLocal() as db_session:
            query = text(
                "SELECT ca.resume_parsed, a.jd_analysis, a.interview_plan, a.role_name, "
                "COALESCE(r.company_name, '') AS company_name "
                "FROM candidate_assessments ca "
                "JOIN assessments a ON ca.assessment_id = a.id "
                "LEFT JOIN recruiters r ON a.recruiter_id = r.id "
                "WHERE ca.id = :ca_id"
            )
            result = await db_session.execute(query, {"ca_id": ca_uuid})
            row = result.fetchone()
            return tuple(row) if row else None
    except Exception:
        logger.exception("Failed to fetch assessment context for CA ID %s", ca_uuid)
        return None
