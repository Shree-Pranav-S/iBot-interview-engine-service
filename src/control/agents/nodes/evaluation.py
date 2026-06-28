"""Simple weak/adequate/strong live evaluation for substantial answers."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.control.agents.nodes.llm_helpers import evaluate_with_schema
from src.control.agents.prompts import LIVE_EVALUATION_SYSTEM_PROMPT
from src.control.agents.state import InterviewState
from src.schemas.prompts import AnswerEvaluationResponse
from src.utils.interview_graph import utc_now_iso

logger = logging.getLogger(__name__)


def _evaluation_messages(state: InterviewState) -> list[dict[str, str]]:
    context = {
        "previous_candidate_response": state.get("previous_candidate_response") or "",
        "previous_question": state.get("current_question_text") or "",
        "current_technical_skill": state.get("current_technical_skill"),
        "expected_signals": list(state.get("current_expected_signals") or []),
        "question_difficulty": state.get("current_question_difficulty"),
        "resume_context": state.get("resume_context")
        or {
            "skills": [],
            "experience_years": 0,
        },
        "schema": AnswerEvaluationResponse.model_json_schema(),
    }
    return [
        {"role": "system", "content": LIVE_EVALUATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, default=str),
        },
    ]


async def evaluate_substantial_answer(state: InterviewState) -> dict[str, Any]:
    """Evaluate one answer and retain only the phase-one live result."""

    source = "llm"
    try:
        result = await evaluate_with_schema(
            _evaluation_messages(state),
            AnswerEvaluationResponse,
        )
    except Exception:
        logger.exception(
            "Strict live answer evaluation failed",
            extra={
                "candidate_assessment_id": state.get("candidate_assessment_id"),
                "question_id": state.get("current_question_id"),
            },
        )
        # A neutral valid result is safer than inventing a negative assessment.
        result = AnswerEvaluationResponse(
            strength="adequate",
            reason="Neutral fallback because the live evaluation service failed.",
        )
        source = "validated_fallback"

    evaluated_at = utc_now_iso()
    skill_streaks = {
        key: dict(value)
        for key, value in (state.get("skill_evaluation_streaks") or {}).items()
    }
    if state.get("current_section_kind") == "technical" and state.get(
        "current_technical_skill"
    ):
        skill_key = str(state["current_technical_skill"]).casefold()
        current = dict(
            skill_streaks.get(skill_key) or {"weak": 0, "adequate": 0, "strong": 0}
        )
        for strength in ("weak", "adequate", "strong"):
            current[strength] = (
                int(current.get(strength) or 0) + 1
                if strength == result.strength
                else 0
            )
        skill_streaks[skill_key] = current

    pending_candidate_turn = dict(state.get("pending_candidate_turn") or {})
    metadata = dict(pending_candidate_turn.get("metadata") or {})
    metadata.update(
        {
            "live_evaluation": result.model_dump(),
            "evaluation_source": source,
            "evaluated_at": evaluated_at,
        }
    )
    pending_candidate_turn["metadata"] = metadata
    return {
        "latest_evaluation": result.model_dump(),
        "evaluation_source": source,
        "last_evaluated_at": evaluated_at,
        "last_answer_strength": result.strength,
        "skill_evaluation_streaks": skill_streaks,
        "pending_candidate_turn": pending_candidate_turn,
        "next_action": "check_time",
    }
