"""Fast live technical evaluation node."""

from __future__ import annotations

from typing import Any

from src.control.agents.nodes.llm_helpers import (
    compact_json,
    live_evaluate_json,
    model_to_dict,
)
from src.control.agents.prompts import EVALUATION_SYSTEM_PROMPT
from src.control.agents.state import InterviewState
from src.schemas.prompts import AnswerEvaluationResponse
from src.utils.interview_graph import utc_now_iso

STRENGTH_SCORE = {"weak": 2.0, "adequate": 3.4, "strong": 4.6}


def _candidate_turn_number(state: InterviewState) -> int:
    pending_candidate = state.get("pending_candidate_turn")
    if pending_candidate:
        return int(pending_candidate.get("turn_number") or 0)
    return int(state.get("turn_number") or 0)


def _skill_key(state: InterviewState) -> str:
    return str(state.get("current_skill") or state.get("current_section") or "").lower()


def _prior_consecutive_adequate(state: InterviewState) -> int:
    progress = dict((state.get("skill_progress") or {}).get(_skill_key(state)) or {})
    return int(progress.get("consecutive_adequate_answers") or 0)


def _recommended_difficulty(state: InterviewState, strength: str) -> str:
    if strength == "weak":
        return "easy"
    if strength == "strong":
        return "hard"
    return "hard" if _prior_consecutive_adequate(state) >= 1 else "medium"


def _evaluation_messages(state: InterviewState) -> list[dict[str, str]]:
    event = state.get("normalized_candidate_event") or {}
    context = {
        "question": state.get("current_question_text"),
        "candidate_answer": event.get("text") or "",
        "section": state.get("current_section"),
        "skill": state.get("current_skill"),
        "difficulty": state.get("current_difficulty"),
    }
    return [
        {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Return JSON matching this schema: "
                f"{compact_json(AnswerEvaluationResponse.model_json_schema(), max_chars=1200)}. "
                f"Context: {compact_json(context, max_chars=2600)}"
            ),
        },
    ]


async def live_evaluate_answer_node(state: InterviewState) -> dict[str, Any]:
    generated = model_to_dict(
        await live_evaluate_json(
            _evaluation_messages(state),
            AnswerEvaluationResponse,
        )
    )
    strength = str(generated["strength"]).lower()
    if strength not in STRENGTH_SCORE:
        raise ValueError(f"Invalid answer strength from evaluator: {strength}")

    recommended_difficulty = _recommended_difficulty(state, strength)
    evaluation = {
        "evaluation_id": (
            f"{state.get('current_question_id')}:{_candidate_turn_number(state)}"
        ),
        "question_id": state.get("current_question_id"),
        "skill": state.get("current_skill"),
        "section": state.get("current_section"),
        "response_type": "answer",
        "provisional_score": STRENGTH_SCORE[strength],
        "strength": strength,
        "is_substantial": True,
        "signals_observed": [f"{strength}_technical_signal"],
        "signals_missing": [] if strength == "strong" else ["depth_or_specificity"],
        "recommended_next_action": "next_question",
        "recommended_difficulty": recommended_difficulty,
        "summary": f"Live evaluation classified the answer as {strength}.",
        "created_at": utc_now_iso(),
        "violations": [],
    }
    return {
        "latest_evaluation": evaluation,
        "live_evaluations": [*(state.get("live_evaluations") or []), evaluation],
        "current_difficulty": recommended_difficulty,
        "violation_to_persist": None,
        "next_node": None,
    }
