"""Orchestrate one-shot DeepSeek evaluation and deterministic persistence."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from src.core.services.evaluation_context_builder import (
    build_evaluation_context,
)
from src.core.services.evaluation_errors import (
    EvaluationNotReadyError,
    PermanentEvaluationError,
)
from src.core.services.evaluation_llm_client import (
    run_deepseek_holistic_evaluation,
)
from src.core.services.evaluation_score_calculator import (
    calculate_final_evaluation,
)
from src.core.services.realtime_event_service import publish_recruiter_event
from src.data.repositories.evaluation_repository import (
    evaluation_exists_for_hash,
    load_evaluation_source,
    save_final_evaluation,
)
from src.schemas.realtime import RecruiterEventType

logger = logging.getLogger(__name__)
READY_SESSION_STATUSES = {
    "COMPLETED",
    "EVALUATED",
    "EVALUATION_FAILED",
}


async def run_holistic_evaluation(
    candidate_assessment_id: str | uuid.UUID,
) -> dict[str, Any]:
    """Load, evaluate, score, persist, and rank one completed interview."""

    candidate_id = str(candidate_assessment_id)
    logger.info(
        "Holistic evaluation pipeline STARTED",
        extra={"candidate_assessment_id": candidate_id},
    )

    source = await load_evaluation_source(candidate_id)
    if source is None:
        raise PermanentEvaluationError(
            f"Evaluation context not found for {candidate_id}"
        )
    session_status = str(source.get("session_status") or "").upper()
    if session_status not in READY_SESSION_STATUSES:
        raise EvaluationNotReadyError(
            f"Interview session is not finalized: {session_status or 'UNKNOWN'}"
        )

    logger.info(
        "Evaluation source loaded, building context",
        extra={
            "candidate_assessment_id": candidate_id,
            "session_status": session_status,
        },
    )

    bundle = build_evaluation_context(source)
    if await evaluation_exists_for_hash(
        candidate_id,
        bundle.evaluation_input.transcript_hash,
    ):
        logger.info(
            "Skipping duplicate holistic evaluation for unchanged context",
            extra={
                "candidate_assessment_id": candidate_id,
                "transcript_hash": bundle.evaluation_input.transcript_hash,
            },
        )
        return {
            "candidate_assessment_id": candidate_id,
            "transcript_hash": bundle.evaluation_input.transcript_hash,
            "status": "already_evaluated",
        }

    logger.info(
        "Calling DeepSeek LLM for holistic evaluation",
        extra={
            "candidate_assessment_id": candidate_id,
            "transcript_hash": bundle.evaluation_input.transcript_hash,
        },
    )

    model_result = await run_deepseek_holistic_evaluation(bundle)

    logger.info(
        "DeepSeek LLM returned, calculating final scores",
        extra={"candidate_assessment_id": candidate_id},
    )

    final_record = calculate_final_evaluation(
        bundle=bundle,
        model_result=model_result,
    )
    notification = await save_final_evaluation(
        final_record,
        recruiter_email=str(source.get("recruiter_email") or ""),
    )
    await publish_recruiter_event(
        recruiter_id=str(source["recruiter_id"]),
        event_type=RecruiterEventType.INTERVIEW_EVALUATED,
        payload={
            "candidate_assessment_id": candidate_id,
            "assessment_id": str(final_record.assessment_id),
            "candidate_name": str(source.get("candidate_name") or "Candidate"),
            "assessment_title": str(source.get("assessment_title") or "Assessment"),
            "overall_score": final_record.overall_score,
            "hiring_recommendation": final_record.hiring_recommendation,
            "status": "EVALUATED",
            "notification_id": str(notification["id"]),
            "notification_sent_at": notification["sent_at"].isoformat(),
        },
    )
    logger.info(
        "Holistic interview evaluation saved",
        extra={
            "candidate_assessment_id": candidate_id,
            "overall_score": final_record.overall_score,
            "hiring_recommendation": (final_record.hiring_recommendation),
            "transcript_hash": final_record.transcript_hash,
        },
    )
    return {
        "candidate_assessment_id": candidate_id,
        "transcript_hash": final_record.transcript_hash,
        "overall_score": final_record.overall_score,
        "hiring_recommendation": final_record.hiring_recommendation,
        "status": "evaluated",
    }
