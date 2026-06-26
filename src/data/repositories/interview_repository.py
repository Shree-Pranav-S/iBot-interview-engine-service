import json
import logging
import uuid
from typing import Any

from sqlalchemy import text

from src.data.clients.postgres_client import get_session_factory

logger = logging.getLogger(__name__)


async def get_interview_data_for_evaluation(
    candidate_assessment_id: uuid.UUID,
) -> dict[str, Any] | None:
    """Fetch durable session JSON plus JD context used for holistic assessment."""
    try:
        SessionLocal = await get_session_factory()
        async with SessionLocal() as db_session:
            result = await db_session.execute(
                text(
                    """
                    SELECT
                        s.id,
                        s.candidate_assessment_id,
                        s.status,
                        s.transcript,
                        s.violations,
                        s.total_elapsed_secs,
                        s.total_pause_secs,
                        ca.assessment_id,
                        ca.resume_parsed,
                        ca.status AS candidate_assessment_status,
                        c.full_name AS candidate_name,
                        c.email AS candidate_email,
                        a.title AS assessment_title,
                        a.role_name,
                        a.jd_text,
                        a.jd_analysis,
                        a.focus_areas,
                        a.interview_plan,
                        a.interview_duration_mins,
                        COALESCE(r.company_name, '') AS company_name
                    FROM interview_sessions s
                    JOIN candidate_assessments ca
                        ON ca.id = s.candidate_assessment_id
                    JOIN candidates c ON c.id = ca.candidate_id
                    JOIN assessments a ON a.id = ca.assessment_id
                    LEFT JOIN recruiters r ON r.id = a.recruiter_id
                    WHERE s.candidate_assessment_id = :candidate_assessment_id
                    """
                ),
                {"candidate_assessment_id": candidate_assessment_id},
            )
            row = result.mappings().first()
            if not row:
                return None

            data = dict(row)
            session = {
                "id": data["id"],
                "candidate_assessment_id": data["candidate_assessment_id"],
                "status": data["status"],
                "transcript": data.get("transcript") or [],
                "violations": data.get("violations") or [],
                "total_elapsed_secs": data.get("total_elapsed_secs") or 0,
                "total_pause_secs": data.get("total_pause_secs") or 0,
            }
            context = {
                "assessment_id": data.get("assessment_id"),
                "candidate_name": data.get("candidate_name"),
                "candidate_email": data.get("candidate_email"),
                "assessment_title": data.get("assessment_title"),
                "role_name": data.get("role_name"),
                "company_name": data.get("company_name"),
                "jd_text": data.get("jd_text"),
                "jd_analysis": data.get("jd_analysis"),
                "focus_areas": data.get("focus_areas"),
                "interview_plan": data.get("interview_plan"),
                "interview_duration_mins": data.get("interview_duration_mins"),
                "resume_parsed": data.get("resume_parsed"),
                "candidate_assessment_status": data.get("candidate_assessment_status"),
            }
            return {"session": session, "context": context}
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
            await db_session.execute(
                text(
                    """
                    UPDATE interview_sessions
                    SET status = 'EVALUATED',
                        last_updated_at = NOW()
                    WHERE id = :session_id
                    """
                ),
                {"session_id": session_id},
            )
            await db_session.execute(
                text(
                    """
                    UPDATE candidate_assessments
                    SET status = 'EVALUATED',
                        interview_ended_at = COALESCE(interview_ended_at, NOW()),
                        updated_at = NOW()
                    WHERE id = :candidate_assessment_id
                    """
                ),
                {"candidate_assessment_id": candidate_assessment_id},
            )
            await db_session.execute(
                text(
                    """
                    WITH assessment_scope AS (
                        SELECT assessment_id
                        FROM candidate_assessments
                        WHERE id = :candidate_assessment_id
                    ),
                    ranked AS (
                        SELECT
                            ie.id,
                            RANK() OVER (
                                ORDER BY ie.overall_score DESC, ie.generated_at ASC
                            ) AS rank_position,
                            COUNT(*) OVER () AS total_count
                        FROM interview_evaluations ie
                        JOIN candidate_assessments ca
                            ON ca.id = ie.candidate_assessment_id
                        WHERE ca.assessment_id = (
                            SELECT assessment_id FROM assessment_scope
                        )
                    )
                    UPDATE interview_evaluations ie
                    SET rank_in_assessment = ranked.rank_position,
                        total_candidates_evaluated = ranked.total_count,
                        percentile_in_assessment = CASE
                            WHEN ranked.total_count <= 1 THEN 100
                            ELSE ROUND(
                                ((ranked.total_count - ranked.rank_position)::numeric
                                / (ranked.total_count - 1)) * 100
                            )::integer
                        END
                    FROM ranked
                    WHERE ie.id = ranked.id
                    """
                ),
                {"candidate_assessment_id": candidate_assessment_id},
            )
            await db_session.commit()
    except Exception:
        logger.exception(
            "Failed to save holistic evaluation for %s", candidate_assessment_id
        )
        raise
