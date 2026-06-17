import logging
import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from src.data.clients.postgres_client import get_session_factory
from src.data.models.postgres.answer_evaluation import AnswerEvaluation
from src.data.models.postgres.interview_evaluation import InterviewEvaluation
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
    """
    try:
        ca_id_str = state.get("candidate_assessment_id")
        if not ca_id_str:
            return

        ca_id = uuid.UUID(ca_id_str)

        # Build session dictionary from state
        sections = state.get("sections", [])
        current_idx = state.get("current_section_index", 0)
        current_section_name = (
            sections[current_idx]["name"]
            if current_idx < len(sections)
            else "self_intro"
        )

        section_progress = {}
        for sec in sections:
            section_progress[sec["name"]] = {
                "time_budget_secs": sec.get("time_budget_secs", 0),
                "time_elapsed_secs": sec.get("time_elapsed_secs", 0),
                "questions_asked": sec.get("questions_asked", 0),
                "concepts_covered": sec.get("concepts_covered", []),
                "is_complete": sec.get("is_complete", False),
            }

        session_data = {
            "status": state.get("session_status", "INITIALIZING").upper(),
            "current_section": current_section_name,
            "current_section_index": current_idx,
            "section_progress": section_progress,
            "current_question_text": state.get("current_question_text"),
            "current_difficulty": state.get("current_question_difficulty", "easy"),
            "consecutive_strong_answers": state.get("consecutive_strong", 0),
            "consecutive_weak_answers": state.get("consecutive_weak", 0),
            "used_concepts": state.get("used_concepts", []),
            "silence_attempt": state.get("silence_attempt", 0),
            "irrelevant_strike_count": state.get("irrelevant_strike_count", 0),
            "timer_started_at": _parse_datetime(state.get("timer_started_at")),
            "total_elapsed_secs": state.get("total_elapsed_secs", 0),
            "total_pause_secs": state.get("total_pause_secs", 0),
            "paused_at": _parse_datetime(state.get("paused_at")),
            "grace_period_expires_at": _parse_datetime(
                state.get("grace_period_expires_at")
            ),
            "auto_submit_triggered": state.get("auto_submit_triggered", False),
        }

        SessionLocal = await get_session_factory()
        async with SessionLocal() as db_session:
            # 1. Upsert InterviewSession
            insert_stmt = insert(InterviewSession).values(
                candidate_assessment_id=ca_id, **session_data
            )
            # Update all fields on conflict
            update_dict = {
                c.name: c
                for c in insert_stmt.excluded
                if c.name not in ["id", "candidate_assessment_id", "created_at"]
            }
            upsert_stmt = insert_stmt.on_conflict_do_update(
                index_elements=["candidate_assessment_id"], set_=update_dict
            ).returning(InterviewSession.id)

            result = await db_session.execute(upsert_stmt)
            session_id = result.scalar_one()

            # 2. Upsert TranscriptTurns
            transcript_turns = state.get("transcript_turns", [])
            for turn in transcript_turns:
                insert(TranscriptTurn).values(
                    candidate_assessment_id=ca_id,
                    session_id=session_id,
                    turn_number=turn.get("turn_number", 0),
                    speaker=turn.get("speaker", "bot"),
                    text=turn.get("text", ""),
                    section=turn.get("section", "unknown"),
                    turn_type=turn.get("turn_type"),
                    response_classification=turn.get("response_classification"),
                    difficulty_at_time=turn.get("difficulty_at_time"),
                    stt_confidence=turn.get("stt_confidence"),
                    concept_tags=turn.get("concept_tags"),
                )

                # Using turn_number and session_id as logical unique constraint for updates
                # Although TranscriptTurn might only have id as PK, we'll just insert if it's new
                # To avoid duplicates, we can query existing turns first
                pass

            # Since TranscriptTurn doesn't have a unique constraint on (session_id, turn_number, speaker)
            # We'll just delete existing and re-insert or query and insert missing
            # Better: query max turn_number and insert only new ones
            existing_turns_q = await db_session.execute(
                select(TranscriptTurn.turn_number, TranscriptTurn.speaker).where(
                    TranscriptTurn.session_id == session_id
                )
            )
            existing_turns = {(row[0], row[1]) for row in existing_turns_q.all()}

            new_turns = []
            for turn in transcript_turns:
                key = (turn.get("turn_number", 0), turn.get("speaker", "bot"))
                if key not in existing_turns:
                    new_turns.append(
                        {
                            "candidate_assessment_id": ca_id,
                            "session_id": session_id,
                            "turn_number": turn.get("turn_number", 0),
                            "speaker": turn.get("speaker", "bot"),
                            "text": turn.get("text", ""),
                            "section": turn.get("section", "unknown"),
                            "turn_type": turn.get("turn_type"),
                            "response_classification": turn.get(
                                "response_classification"
                            ),
                            "difficulty_at_time": turn.get("difficulty_at_time"),
                            "stt_confidence": turn.get("stt_confidence"),
                            "concept_tags": turn.get("concept_tags"),
                        }
                    )
            if new_turns:
                await db_session.execute(insert(TranscriptTurn).values(new_turns))

            # 3. Upsert AnswerEvaluations
            answer_evals = state.get("answer_evaluations", [])
            existing_evals_q = await db_session.execute(
                select(AnswerEvaluation.turn_number).where(
                    AnswerEvaluation.session_id == session_id
                )
            )
            existing_evals = {row[0] for row in existing_evals_q.all()}

            new_evals = []
            for eval_data in answer_evals:
                if eval_data.get("turn_number") not in existing_evals:
                    new_evals.append(
                        {
                            "candidate_assessment_id": ca_id,
                            "session_id": session_id,
                            "turn_number": eval_data.get("turn_number", 0),
                            "section": eval_data.get("section", "unknown"),
                            "skill": eval_data.get("skill"),
                            "question_text": eval_data.get("question_text", ""),
                            "answer_text": eval_data.get("answer_text", ""),
                            "response_classification": eval_data.get(
                                "response_classification", "answer"
                            ),
                            "quality": eval_data.get("quality", "adequate"),
                            "score": eval_data.get("score", 0.0),
                            "difficulty_at_time": eval_data.get(
                                "difficulty_at_time", "easy"
                            ),
                            "nudge_given": eval_data.get("nudge_given", False),
                            "signals_present": eval_data.get("signals_present", []),
                            "signals_missing": eval_data.get("signals_missing", []),
                            "tone_scores": eval_data.get("tone_scores"),
                            "one_line_feedback": eval_data.get("one_line_feedback"),
                        }
                    )
            if new_evals:
                await db_session.execute(insert(AnswerEvaluation).values(new_evals))

            await db_session.commit()
    except Exception:
        logger.exception("Failed to sync interview state to Postgres.")


async def get_interview_data_for_evaluation(
    candidate_assessment_id: uuid.UUID,
) -> dict | None:
    """Fetch session, transcript turns, and evaluations for holistic assessment."""
    try:
        SessionLocal = await get_session_factory()
        async with SessionLocal() as db_session:
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
        SessionLocal = await get_session_factory()
        async with SessionLocal() as db_session:
            stmt = insert(InterviewEvaluation).values(
                candidate_assessment_id=candidate_assessment_id,
                session_id=session_id,
                **eval_data,
            )
            update_dict = {
                c.name: c
                for c in stmt.excluded
                if c.name
                not in ["id", "candidate_assessment_id", "session_id", "generated_at"]
            }
            stmt = stmt.on_conflict_do_update(
                index_elements=["candidate_assessment_id"], set_=update_dict
            )
            await db_session.execute(stmt)
            await db_session.commit()
            logger.info(
                f"Successfully saved holistic evaluation for {candidate_assessment_id}"
            )
    except Exception:
        logger.exception(
            f"Failed to save holistic evaluation for {candidate_assessment_id}"
        )
