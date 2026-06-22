import logging
import uuid
from datetime import datetime

from sqlalchemy import select, text

from src.data.clients.postgres_client import get_db_session
from src.data.models.postgres.answer_evaluation import AnswerEvaluation
from src.data.models.postgres.interview_session import InterviewSession
from src.data.models.postgres.transcript_turn import TranscriptTurn

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
    relational schema (InterviewSession, TranscriptTurn, AnswerEvaluation).
    Disabled/made a no-op as the database schema has changed.
    """
    logger.debug("sync_interview_state skipped (no-op)")


async def get_interview_data_for_evaluation(
    candidate_assessment_id: uuid.UUID,
) -> dict | None:
    """Fetch session, transcript turns, and evaluations for holistic assessment."""
    try:
        async for db_session in get_db_session():
            # Get session
            stmt = select(InterviewSession).where(
                InterviewSession.candidate_assessment_id == candidate_assessment_id
            )
            result = await db_session.execute(stmt)
            session = result.scalar_one_or_none()

            if not session:
                return None

            # Get transcript turns
            stmt_turns = (
                select(TranscriptTurn)
                .where(TranscriptTurn.session_id == session.id)
                .order_by(TranscriptTurn.turn_number)
            )
            turns_result = await db_session.execute(stmt_turns)
            turns = turns_result.scalars().all()

            # Get answer evaluations
            stmt_evals = (
                select(AnswerEvaluation)
                .where(AnswerEvaluation.session_id == session.id)
                .order_by(AnswerEvaluation.turn_number)
            )
            evals_result = await db_session.execute(stmt_evals)
            evals = evals_result.scalars().all()

            return {"session": session, "turns": turns, "evaluations": evals}
        return None
    except Exception:
        logger.exception(
            f"Failed to fetch interview data for {candidate_assessment_id}"
        )
        return None


async def save_holistic_evaluation(
    candidate_assessment_id: uuid.UUID, session_id: uuid.UUID, eval_data: dict
) -> None:
    """Saves the holistic evaluation JSON output to the Postgres table."""
    try:
        from src.data.models.postgres.interview_evaluation import InterviewEvaluation

        async for db_session in get_db_session():
            evaluation = InterviewEvaluation(
                candidate_assessment_id=candidate_assessment_id,
                session_id=session_id,
                skill_scores=eval_data.get("skill_scores", {}),
                technical_dimension_score=eval_data.get(
                    "technical_dimension_score", 0.0
                ),
                problem_solving_score=eval_data.get("problem_solving_score", 0.0),
                problem_solving_evidence=eval_data.get("problem_solving_evidence", []),
                problem_solving_summary=eval_data.get("problem_solving_summary", ""),
                communication_score=eval_data.get("communication_score", 0.0),
                communication_evidence=eval_data.get("communication_evidence", []),
                communication_summary=eval_data.get("communication_summary", ""),
                behavioural_score=eval_data.get("behavioural_score", 0.0),
                behavioural_evidence=eval_data.get("behavioural_evidence", []),
                behavioural_summary=eval_data.get("behavioural_summary", ""),
                cultural_fit_score=eval_data.get("cultural_fit_score", 0.0),
                cultural_fit_evidence=eval_data.get("cultural_fit_evidence", []),
                cultural_fit_summary=eval_data.get("cultural_fit_summary", ""),
                section_summaries=eval_data.get("section_summaries", {}),
                overall_score=eval_data.get("overall_score", 0.0),
                hiring_recommendation=eval_data.get(
                    "hiring_recommendation", "CONSIDER"
                ),
                overall_narrative=eval_data.get("overall_narrative", ""),
                strengths=eval_data.get("strengths", []),
                concerns=eval_data.get("concerns", []),
                red_flags=eval_data.get("red_flags", []),
                violation_summary=eval_data.get("violation_summary"),
                best_answer=eval_data.get("best_answer"),
                weakest_answer=eval_data.get("weakest_answer"),
                recommendation_reasoning=eval_data.get("recommendation_reasoning", ""),
            )
            db_session.add(evaluation)
            await db_session.commit()
            return
    except Exception:
        logger.exception(
            f"Failed to save holistic evaluation for {candidate_assessment_id}"
        )


async def get_assessment_context_by_ca_id(ca_uuid: uuid.UUID) -> tuple | None:
    """Fetches resume, JD analysis, interview plan, and role name for an assessment."""
    try:
        async for db_session in get_db_session():
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
        return None
    except Exception:
        logger.exception(f"Failed to fetch assessment context for CA ID {ca_uuid}")
        return None
