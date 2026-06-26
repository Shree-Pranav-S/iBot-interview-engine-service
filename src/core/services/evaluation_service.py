"""Final holistic interview evaluation service."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from pydantic import ValidationError

from src.core.services.llm_service import evaluate
from src.data.repositories.interview_repository import (
    get_interview_data_for_evaluation,
    save_holistic_evaluation,
)
from src.schemas.evaluation import HolisticEvaluationReport
from src.utils.evaluation import (
    _build_model_audit_context,
    _build_records_text,
    _build_transcript_text,
    _candidate_answer_records,
    _is_candidate_turn,
)
from src.utils.holistic_evaluation_prompt import HOLISTIC_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def _parse_model_report(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Holistic evaluation response must be a JSON object")
    report = HolisticEvaluationReport.model_validate(parsed)
    return report.model_dump(mode="json")


def _json_blob(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, default=str)


async def _generate_valid_report(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Call the 70B evaluator, then retry once with validation feedback if needed."""

    response_json = await evaluate(messages)
    try:
        return _parse_model_report(response_json)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        repair_messages = [
            *messages,
            {
                "role": "assistant",
                "content": response_json[:12000],
            },
            {
                "role": "user",
                "content": (
                    "The previous holistic evaluation JSON failed validation. "
                    "Return a corrected complete JSON object only, matching the "
                    "schema exactly with no extra keys. Preserve the same evidence-"
                    "backed conclusions where possible. Validation error: "
                    f"{str(exc)[:2400]}"
                ),
            },
        ]
        repaired_json = await evaluate(repair_messages)
        return _parse_model_report(repaired_json)


async def run_holistic_evaluation(
    candidate_assessment_id: str,
    state: dict[str, Any] | None = None,
) -> bool:
    """Generate and persist the final evaluation for a candidate assessment."""
    try:
        ca_uuid = uuid.UUID(candidate_assessment_id)
    except ValueError:
        logger.error("Invalid UUID for evaluation: %s", candidate_assessment_id)
        return False

    state = state or {}
    data = await get_interview_data_for_evaluation(ca_uuid)
    session = (data or {}).get("session") or {}
    context = (data or {}).get("context") or {}

    session_id_value = (
        session.get("id")
        or state.get("interview_session_id")
        or state.get("session_id")
    )
    if not session_id_value:
        logger.warning("No interview session found for %s", candidate_assessment_id)
        return False

    session_id = uuid.UUID(str(session_id_value))
    transcript = list(session.get("transcript") or state.get("transcript") or [])
    violations = list(session.get("violations") or state.get("violations") or [])
    session_status = str(session.get("status") or state.get("session_status") or "")

    if not any(
        _is_candidate_turn(turn) for turn in transcript if isinstance(turn, dict)
    ):
        logger.warning(
            "No candidate transcript turns found for %s", candidate_assessment_id
        )
        return False

    live_evaluations = list(state.get("live_evaluations") or [])
    records = _candidate_answer_records(transcript, live_evaluations)
    audit_context = _build_model_audit_context(
        records=records,
        violations=violations,
        context=context,
        session_status=session_status,
    )

    user_message = (
        "Return JSON matching this exact Pydantic JSON schema:\n"
        f"{json.dumps(HolisticEvaluationReport.model_json_schema(), ensure_ascii=True)}\n\n"
        "This final evaluation runs after the live interview, so optimize for "
        "accuracy, evidence coverage, and detailed recruiter usefulness rather "
        "than speed. The final evaluator is expected to use the configured 70B "
        "Groq evaluation model.\n\n"
        "Canonical transcript loaded from interview_sessions.transcript. This is the "
        "primary evidence source for candidate performance:\n"
        f"{_build_transcript_text(transcript)}\n\n"
        "Canonical transcript JSON with all stored metadata. Parse every candidate "
        "and interviewer turn, including section, skill, difficulty, question_id, "
        "turn_number, and response metadata:\n"
        f"{_json_blob(transcript)}\n\n"
        "Live answer-evaluation signals from the graph. Treat these as supporting "
        "context only. You may use them to understand provisional scoring and observed/"
        "missing signals, but the final report must still be evidence-backed by the "
        "canonical transcript and recorded violations:\n"
        f"{_build_records_text(records)}\n\n"
        "Full structured candidate answer records with associated question context "
        "and live signals. Use these to align each answer to its question, section, "
        "skill, and difficulty, but re-judge quality from the transcript yourself:\n"
        f"{_json_blob(records)}\n\n"
        "Candidate, assessment, JD analysis, skill priorities, resume context, "
        "interview plan, and focus-area overrides. Use resume context as background "
        "only, not as proof of demonstrated ability:\n"
        f"{_json_blob(context)}\n\n"
        "Session violations loaded from interview_sessions.violations. These are "
        "authoritative recorded violations and must be reflected in the report when present:\n"
        f"{_json_blob(violations) if violations else 'None'}\n\n"
        "Computed audit context for cross-checking only. Do not copy it blindly. Use it "
        "to verify weighting, skill coverage, best/weakest answer hints, violation penalty "
        "hints, and section summaries, but produce the final report yourself from the "
        "transcript, violations, JD/interview plan, and live signals:\n"
        f"{_json_blob(audit_context)}"
    )

    messages = [
        {"role": "system", "content": HOLISTIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        eval_data = await _generate_valid_report(messages)
    except Exception:
        logger.exception(
            "Failed to generate or validate holistic evaluation for %s",
            ca_uuid,
        )
        return False

    try:
        await save_holistic_evaluation(ca_uuid, session_id, eval_data)
    except Exception:
        logger.exception("Failed to persist holistic evaluation for %s", ca_uuid)
        return False

    logger.info("Holistic evaluation completed and saved for %s", ca_uuid)
    return True
