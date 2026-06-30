"""Simple weak/adequate/strong live evaluation for substantial answers."""

from __future__ import annotations

import json
import logging
from typing import Any

from src.control.agents.key_routing import active_turn_key_slot
from src.control.agents.nodes.llm_helpers import evaluate_with_schema
from src.control.agents.prompts import LIVE_EVALUATION_SYSTEM_PROMPT
from src.control.agents.state import InterviewState
from src.schemas.prompts import AnswerEvaluationResponse
from src.utils.interview_graph import utc_now_iso

logger = logging.getLogger(__name__)


def _evaluation_messages(state: InterviewState) -> list[dict[str, str]]:
    """
    Construct the messages to prompt the LLM for a live evaluation of the candidate's answer.

    Args:
        state: The current interview state.

    Returns:
        A list of chat messages containing the evaluation system prompt and the candidate's answer.
    """
    context = {
        "previous_candidate_response": state.get("previous_candidate_response") or "",
        "previous_question": state.get("current_question_text") or "",
    }
    return [
        {"role": "system", "content": LIVE_EVALUATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, default=str),
        },
    ]


async def evaluate_substantial_answer(state: InterviewState) -> dict[str, Any]:
    """
    Evaluate one substantial answer and retain only the phase-one live result.

    This node classifies the response strength as weak, adequate, or strong, and updates
    streak counters for adaptive difficulty tuning in the `question_strategy` node.

    Args:
        state: The current interview state.

    Returns:
        State updates containing the live evaluation result, updated streaks, and routing key.
    """

    source = "llm"
    try:
        result = await evaluate_with_schema(
            _evaluation_messages(state),
            AnswerEvaluationResponse,
            key_slot=active_turn_key_slot(state),
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
        "skill_evaluation_streaks": skill_streaks,
        "pending_candidate_turn": pending_candidate_turn,
        "next_action": "check_time",
    }
