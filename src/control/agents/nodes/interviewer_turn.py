"""Two-stage candidate routing and live interviewer response handling.

High-confidence deterministic paths remain local. Other utterances first receive a
small classification-only call. A second model call runs only when a substantial
answer needs evaluation/question generation or a dynamic clarification is required.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from src.control.agents.key_routing import active_turn_key_slot
from src.control.agents.state import InterviewState
from src.control.agents.utils.classify_response import (
    BEHAVIOURAL_ANSWER_WORD_THRESHOLD,
    BEHAVIOURAL_SUBSTANTIAL_WORD_THRESHOLD,
    _deterministic_classification,
    _spoken_word_count,
    self_intro_is_substantial,
)
from src.control.agents.utils.interviewer_turn import (
    _apply_result,
    _classification_messages,
    _difficulty_plan_for_target,
    _finish_classification_only,
    _finish_deterministic,
    _guard_against_overstrict_irrelevance,
    _live_interviewer_messages,
    _resolve_target,
    _response_mode,
    _safe_classification_fallback,
    _safe_live_fallback,
    _should_finish_deterministically,
    _validate_live_response,
)
from src.control.agents.utils.question_strategy import is_self_intro_phase
from src.control.agents.utils.time_manager import decide_time_action
from src.core.services import llm_service
from src.schemas.prompts import (
    CandidateResponseClassification,
    LiveInterviewerResponse,
)

logger = logging.getLogger(__name__)


async def interviewer_turn(state: InterviewState) -> dict[str, Any]:
    """Classify first, then conditionally evaluate and generate a live response."""

    started_at = time.perf_counter()
    text = str(state.get("previous_candidate_response") or "")
    det = _deterministic_classification(state, text)

    if det is not None and _should_finish_deterministically(det):
        return _finish_deterministic(state, det, started_at)

    if is_self_intro_phase(state):
        # Preserve the existing cumulative-word deterministic introduction rule.
        deterministic_classification = CandidateResponseClassification(
            response_type="answer",
            clarification_type=None,
            is_substantial=self_intro_is_substantial(state, text),
            interview_meta_type=None,
        )
        decision = decide_time_action(state)
        target = _resolve_target(state, decision)
        plan = _difficulty_plan_for_target(state, target)
        return _apply_result(
            state,
            deterministic_classification,
            None,
            decision,
            target,
            plan,
            "deterministic_self_intro",
            None,
            started_at,
        )

    word_count = _spoken_word_count(text)
    if (
        state.get("current_section_kind") == "behavioural_cultural"
        and word_count > BEHAVIOURAL_ANSWER_WORD_THRESHOLD
    ):
        # Preserve the existing behavioural word-count fast path.
        deterministic_classification = CandidateResponseClassification(
            response_type="answer",
            clarification_type=None,
            is_substantial=(word_count > BEHAVIOURAL_SUBSTANTIAL_WORD_THRESHOLD),
            interview_meta_type=None,
        )
        decision = decide_time_action(state)
        target = _resolve_target(state, decision)
        return _apply_result(
            state,
            deterministic_classification,
            None,
            decision,
            target,
            None,
            "deterministic_behavioural",
            None,
            started_at,
        )

    classification: CandidateResponseClassification | None = None
    classification_source = "llm"
    try:
        classification = await llm_service.classify(
            _classification_messages(state),
            CandidateResponseClassification,
            key_slot=active_turn_key_slot(state),
        )
        guarded = _guard_against_overstrict_irrelevance(
            state,
            text,
            classification,
        )
        if guarded is not classification:
            logger.info(
                "Promoted topical technical response from irrelevant to answer",
                extra={
                    "candidate_assessment_id": state.get("candidate_assessment_id"),
                    "question_id": state.get("current_question_id"),
                },
            )
            classification = guarded
            classification_source = "llm_technical_relevance_guard"
    except Exception:
        logger.exception(
            "Candidate-response classification failed; using safe fallback",
            extra={
                "candidate_assessment_id": state.get("candidate_assessment_id"),
                "question_id": state.get("current_question_id"),
            },
        )
        classification = _safe_classification_fallback(text)
        classification_source = "validated_fallback"

    response_mode = _response_mode(classification)
    if response_mode is None:
        return _finish_classification_only(
            state,
            classification,
            classification_source,
            started_at,
        )

    decision = decide_time_action(state)
    target = _resolve_target(state, decision)
    must_close = decision["action"] == "close"
    ask_next_question = not must_close
    plan = _difficulty_plan_for_target(state, target)

    response_source = "llm"
    messages = _live_interviewer_messages(
        state,
        classification=classification,
        target=target,
        decision=decision,
        ask_next_question=ask_next_question,
        must_close=must_close,
        plan=plan,
    )
    result: LiveInterviewerResponse | None
    try:
        result = await llm_service.respond(
            messages,
            LiveInterviewerResponse,
            key_slot=active_turn_key_slot(state),
        )
        _validate_live_response(classification, state, result)
    except Exception:
        logger.exception(
            "Live evaluation/interviewer call failed; using local fallback",
            extra={
                "candidate_assessment_id": state.get("candidate_assessment_id"),
                "question_id": state.get("current_question_id"),
                "response_mode": response_mode,
            },
        )
        result = _safe_live_fallback(
            response_mode,
            is_technical_answer=(state.get("current_section_kind") == "technical"),
        )
        response_source = "validated_fallback"

    return _apply_result(
        state,
        classification,
        result,
        decision,
        target,
        plan,
        classification_source,
        response_source,
        started_at,
    )
