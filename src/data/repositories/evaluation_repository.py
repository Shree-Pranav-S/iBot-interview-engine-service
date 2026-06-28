"""Database operations for one-shot holistic interview evaluation."""

from __future__ import annotations

import json
import uuid
from typing import Any

from sqlalchemy import text

from src.data.clients.postgres_client import get_session_factory
from src.schemas.evaluation_llm import FinalEvaluationRecord


def _uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


async def load_evaluation_source(
    candidate_assessment_id: str | uuid.UUID,
) -> dict[str, Any] | None:
    """Load full transcript, violations, JD, plan, and candidate context."""

    session_factory = await get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """
                SELECT
                    s.id AS session_id,
                    s.candidate_assessment_id,
                    s.status AS session_status,
                    s.transcript,
                    s.violations,
                    s.total_elapsed_secs,
                    s.total_pause_secs,
                    ca.assessment_id,
                    ca.status AS candidate_assessment_status,
                    a.recruiter_id,
                    c.full_name AS candidate_name,
                    c.email AS candidate_email,
                    a.title AS assessment_title,
                    a.role_name,
                    a.jd_analysis,
                    a.interview_plan,
                    a.interview_duration_mins,
                    COALESCE(r.company_name, '') AS company_name,
                    COALESCE(r.email, '') AS recruiter_email
                FROM interview_sessions s
                JOIN candidate_assessments ca
                    ON ca.id = s.candidate_assessment_id
                JOIN candidates c ON c.id = ca.candidate_id
                JOIN assessments a ON a.id = ca.assessment_id
                LEFT JOIN recruiters r ON r.id = a.recruiter_id
                WHERE s.candidate_assessment_id = :candidate_assessment_id
                """
            ),
            {"candidate_assessment_id": _uuid(candidate_assessment_id)},
        )
        row = result.mappings().first()
        return dict(row) if row else None


async def evaluation_exists_for_hash(
    candidate_assessment_id: str | uuid.UUID,
    transcript_hash: str,
) -> bool:
    """Return true when this exact immutable context was already evaluated."""

    session_factory = await get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM interview_evaluations
                    WHERE candidate_assessment_id = :candidate_assessment_id
                      AND transcript_hash = :transcript_hash
                )
                """
            ),
            {
                "candidate_assessment_id": _uuid(candidate_assessment_id),
                "transcript_hash": transcript_hash,
            },
        )
        return bool(result.scalar())


async def mark_evaluation_failed(
    candidate_assessment_id: str | uuid.UUID,
) -> None:
    """Persist terminal evaluation failure without changing candidate outcome."""

    session_factory = await get_session_factory()
    async with session_factory() as session, session.begin():
        await session.execute(
            text(
                """
                    UPDATE interview_sessions
                    SET status = 'EVALUATION_FAILED',
                        last_updated_at = NOW()
                    WHERE candidate_assessment_id = :candidate_assessment_id
                      AND status <> 'EVALUATED'
                    """
            ),
            {"candidate_assessment_id": _uuid(candidate_assessment_id)},
        )


async def save_final_evaluation(
    record: FinalEvaluationRecord,
    *,
    recruiter_email: str,
) -> dict[str, Any]:
    """Atomically save the report, lifecycle state, ranks, and dashboard notice."""

    params = {
        **record.model_dump(mode="python"),
        "candidate_assessment_id": _uuid(record.candidate_assessment_id),
        "session_id": _uuid(record.session_id),
        "assessment_id": _uuid(record.assessment_id),
        "skill_scores": _json(record.skill_scores),
        "skill_summary": _json(record.skill_summary),
        "skill_evidence": _json(record.skill_evidence),
        "section_communication_scores": _json(record.section_communication_scores),
        "violation_summary": _json(record.violation_summary),
        "raw_model_output": _json(record.raw_model_output),
    }
    session_factory = await get_session_factory()
    async with session_factory() as session, session.begin():
        await session.execute(
            text(
                """
                    INSERT INTO interview_evaluations (
                        candidate_assessment_id,
                        session_id,
                        intro_section_score,
                        intro_section_summary,
                        intro_section_evidence,
                        skill_scores,
                        overall_technical_skill_score,
                        skill_summary,
                        skill_evidence,
                        behavioural_cultural_score,
                        behavioural_cultural_summary,
                        behavioural_cultural_evidence,
                        communication_score,
                        communication_summary,
                        communication_evidence,
                        section_communication_scores,
                        violation_summary,
                        violation_evidence,
                        raw_overall_score,
                        violation_penalty,
                        overall_score,
                        hiring_recommendation,
                        model_recommendation,
                        recommendation_override_reason,
                        overall_summary,
                        recommendation_reasoning,
                        strengths,
                        concerns,
                        prompt_version,
                        model_name,
                        model_provider,
                        evaluation_schema_version,
                        transcript_hash,
                        raw_model_output
                    )
                    VALUES (
                        :candidate_assessment_id,
                        :session_id,
                        :intro_section_score,
                        :intro_section_summary,
                        :intro_section_evidence,
                        CAST(:skill_scores AS jsonb),
                        :overall_technical_skill_score,
                        CAST(:skill_summary AS jsonb),
                        CAST(:skill_evidence AS jsonb),
                        :behavioural_cultural_score,
                        :behavioural_cultural_summary,
                        :behavioural_cultural_evidence,
                        :communication_score,
                        :communication_summary,
                        :communication_evidence,
                        CAST(:section_communication_scores AS jsonb),
                        CAST(:violation_summary AS jsonb),
                        :violation_evidence,
                        :raw_overall_score,
                        :violation_penalty,
                        :overall_score,
                        :hiring_recommendation,
                        :model_recommendation,
                        :recommendation_override_reason,
                        :overall_summary,
                        :recommendation_reasoning,
                        :strengths,
                        :concerns,
                        :prompt_version,
                        :model_name,
                        :model_provider,
                        :evaluation_schema_version,
                        :transcript_hash,
                        CAST(:raw_model_output AS jsonb)
                    )
                    ON CONFLICT (candidate_assessment_id) DO UPDATE SET
                        session_id = EXCLUDED.session_id,
                        intro_section_score = EXCLUDED.intro_section_score,
                        intro_section_summary = EXCLUDED.intro_section_summary,
                        intro_section_evidence = EXCLUDED.intro_section_evidence,
                        skill_scores = EXCLUDED.skill_scores,
                        overall_technical_skill_score =
                            EXCLUDED.overall_technical_skill_score,
                        skill_summary = EXCLUDED.skill_summary,
                        skill_evidence = EXCLUDED.skill_evidence,
                        behavioural_cultural_score =
                            EXCLUDED.behavioural_cultural_score,
                        behavioural_cultural_summary =
                            EXCLUDED.behavioural_cultural_summary,
                        behavioural_cultural_evidence =
                            EXCLUDED.behavioural_cultural_evidence,
                        communication_score = EXCLUDED.communication_score,
                        communication_summary = EXCLUDED.communication_summary,
                        communication_evidence = EXCLUDED.communication_evidence,
                        section_communication_scores =
                            EXCLUDED.section_communication_scores,
                        violation_summary = EXCLUDED.violation_summary,
                        violation_evidence = EXCLUDED.violation_evidence,
                        raw_overall_score = EXCLUDED.raw_overall_score,
                        violation_penalty = EXCLUDED.violation_penalty,
                        overall_score = EXCLUDED.overall_score,
                        hiring_recommendation = EXCLUDED.hiring_recommendation,
                        model_recommendation = EXCLUDED.model_recommendation,
                        recommendation_override_reason =
                            EXCLUDED.recommendation_override_reason,
                        overall_summary = EXCLUDED.overall_summary,
                        recommendation_reasoning =
                            EXCLUDED.recommendation_reasoning,
                        strengths = EXCLUDED.strengths,
                        concerns = EXCLUDED.concerns,
                        prompt_version = EXCLUDED.prompt_version,
                        model_name = EXCLUDED.model_name,
                        model_provider = EXCLUDED.model_provider,
                        evaluation_schema_version =
                            EXCLUDED.evaluation_schema_version,
                        transcript_hash = EXCLUDED.transcript_hash,
                        raw_model_output = EXCLUDED.raw_model_output,
                        generated_at = NOW()
                    """
            ),
            params,
        )
        await session.execute(
            text(
                """
                    UPDATE interview_sessions
                    SET status = 'EVALUATED',
                        last_updated_at = NOW()
                    WHERE id = :session_id
                    """
            ),
            params,
        )
        await session.execute(
            text(
                """
                    UPDATE candidate_assessments
                    SET status = 'EVALUATED',
                        interview_ended_at = COALESCE(
                            interview_ended_at,
                            NOW()
                        ),
                        updated_at = NOW()
                    WHERE id = :candidate_assessment_id
                    """
            ),
            params,
        )
        notification_result = await session.execute(
            text(
                """
                    INSERT INTO notification_logs (
                        candidate_assessment_id,
                        notification_type,
                        recipient_email,
                        delivery_status
                    )
                    VALUES (
                        :candidate_assessment_id,
                        'REPORT_READY',
                        :recruiter_email,
                        'SENT'
                    )
                    RETURNING id, sent_at
                    """
            ),
            {
                **params,
                "recruiter_email": recruiter_email,
            },
        )
        notification = notification_result.mappings().one()

        # Serialize ranking refreshes within an assessment so concurrent
        # evaluations cannot publish inconsistent ranks or percentiles.
        await session.execute(
            text(
                """
                    SELECT pg_advisory_xact_lock(
                        hashtext(:assessment_lock_key)
                    )
                    """
            ),
            {
                "assessment_lock_key": str(record.assessment_id),
            },
        )
        await session.execute(
            text(
                """
                    WITH ranked AS (
                        SELECT
                            ie.id,
                            RANK() OVER (
                                ORDER BY
                                    ie.overall_score DESC,
                                    ie.generated_at ASC
                            ) AS rank_position,
                            COUNT(*) OVER () AS total_count
                        FROM interview_evaluations ie
                        JOIN candidate_assessments ca
                            ON ca.id = ie.candidate_assessment_id
                        WHERE ca.assessment_id = :assessment_id
                    )
                    UPDATE interview_evaluations ie
                    SET rank_in_assessment = ranked.rank_position,
                        total_candidates_evaluated = ranked.total_count,
                        percentile_in_assessment = CASE
                            WHEN ranked.total_count <= 1 THEN 100
                            ELSE ROUND(
                                (
                                    (
                                        ranked.total_count
                                        - ranked.rank_position
                                    )::numeric
                                    / (ranked.total_count - 1)
                                ) * 100
                            )::integer
                        END
                    FROM ranked
                    WHERE ie.id = ranked.id
                    """
            ),
            params,
        )
    return dict(notification)
