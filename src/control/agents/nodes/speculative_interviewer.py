"""Speculative merged interviewer-turn warmup without LangGraph resume."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, cast

from src.control.agents.key_routing import compute_turn_key_slot
from src.control.agents.nodes.classify_response import _deterministic_classification
from src.control.agents.nodes.interviewer_turn import (
    _interviewer_messages,
    _resolve_target,
    _should_finish_deterministically,
)
from src.control.agents.nodes.llm_helpers import interviewer_turn_with_schema
from src.control.agents.nodes.question_diversity import normalize_turn_text
from src.control.agents.nodes.question_strategy import difficulty_plan
from src.control.agents.nodes.time_manager import decide_time_action
from src.control.agents.state import InterviewState
from src.schemas.prompts import InterviewerTurnResponse

logger = logging.getLogger(__name__)
_MIN_WARMUP_WORDS = 3


@dataclass
class SpeculativeCacheEntry:
    """Cached merged interviewer result keyed by normalized candidate text."""

    normalized_text: str
    result: dict[str, Any]
    created_at: float
    state_version: str


def _state_version(state: dict[str, Any]) -> str:
    return "|".join(
        [
            str(state.get("current_question_id") or ""),
            str(len(state.get("asked_questions") or [])),
            str(state.get("current_section_index") or ""),
        ]
    )


async def warm_speculative_interviewer_turn(
    state: dict[str, Any],
    text: str,
) -> SpeculativeCacheEntry | None:
    """
    Run the merged interviewer LLM call without resuming LangGraph.

    Returns None when the utterance is too short or deterministic routing applies.
    """

    normalized = normalize_turn_text(text)
    if len(normalized.split()) < _MIN_WARMUP_WORDS:
        return None

    warm_state = cast(
        InterviewState,
        {
            **state,
            "previous_candidate_response": " ".join(text.split()),
        },
    )
    det = _deterministic_classification(warm_state, text)
    if _should_finish_deterministically(det):
        return None

    decision = decide_time_action(warm_state)
    target = _resolve_target(warm_state, decision)
    must_close = decision["action"] == "close"
    ask_next_question = not must_close
    plan = (
        difficulty_plan(
            warm_state,
            skill=str(
                target["skill"] or warm_state.get("current_technical_skill") or "skill"
            ),
            entering_new_section=bool(target["entering_new_section"]),
        )
        if target["kind"] == "technical"
        else None
    )
    messages = _interviewer_messages(
        warm_state,
        target=target,
        decision=decision,
        ask_next_question=ask_next_question,
        must_close=must_close,
        plan=plan,
    )

    try:
        result = await interviewer_turn_with_schema(
            messages,
            InterviewerTurnResponse,
            key_slot=compute_turn_key_slot(state),
        )
    except Exception:
        logger.warning(
            "Speculative interviewer-turn warmup failed",
            exc_info=True,
            extra={
                "candidate_assessment_id": state.get("candidate_assessment_id"),
            },
        )
        return None

    return SpeculativeCacheEntry(
        normalized_text=normalized,
        result=result.model_dump(),
        created_at=time.monotonic(),
        state_version=_state_version(state),
    )


def cache_matches_final(
    entry: SpeculativeCacheEntry, final_text: str, state: dict[str, Any]
) -> bool:
    """Return True when a speculative cache entry is safe to commit."""

    if entry.state_version != _state_version(state):
        return False
    return entry.normalized_text == normalize_turn_text(final_text)
