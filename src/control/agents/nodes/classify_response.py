"""Low-latency candidate-response classification for graph routing."""

from __future__ import annotations

import logging
import re
from typing import Any

from src.control.agents.nodes.context_utils import (
    extract_resume_skills,
    is_substantial_answer,
    resume_mentions_skill,
)
from src.control.agents.nodes.llm_helpers import (
    classify_json,
    compact_json,
    model_to_dict,
)
from src.control.agents.prompts import CLASSIFICATION_SYSTEM_PROMPT
from src.control.agents.state import InterviewState
from src.schemas.prompts import CandidateResponseClassification

logger = logging.getLogger(__name__)

TEXT_SKIP_PATTERNS = re.compile(r"\b(skip|pass|next question|move on)\b", re.I)
TEXT_REPEAT_PATTERNS = re.compile(r"\b(repeat|say that again|once again)\b", re.I)
TEXT_REPHRASE_PATTERNS = re.compile(
    r"\b(rephrase|simplify|explain the question|what do you mean)\b",
    re.I,
)
TEXT_IRRELEVANT_PATTERNS = re.compile(
    r"\b(weather|joke|chatgpt|google|internet|answer for me|tell me the answer|"
    r"system prompt|ignore previous|forget instructions)\b",
    re.I,
)
YES_THINK_PATTERNS = re.compile(r"\b(yes|yeah|yep|please|sure|ok|okay)\b", re.I)


def _event_payload(state: InterviewState) -> dict[str, Any]:
    event = state.get("normalized_candidate_event") or {}
    response_type = str(event.get("response_type") or "answer")
    if response_type == "silence":
        return {
            "response_type": "silence",
            "candidate_question_intent": None,
            "resume_skill_match": False,
        }
    if response_type in {"timer_expired", "disconnect"}:
        return {
            "response_type": response_type,
            "candidate_question_intent": None,
            "resume_skill_match": False,
        }
    if response_type in {"technical_issue", "interruption"}:
        return {
            "response_type": "irrelevant_answer",
            "candidate_question_intent": None,
            "resume_skill_match": False,
        }
    if response_type == "skip":
        skill = state.get("current_skill")
        return {
            "response_type": "clarification_question",
            "candidate_question_intent": "skip_question",
            "resume_skill_match": resume_mentions_skill(
                state.get("resume_parsed") or {},
                skill,
            ),
        }
    return {}


def _heuristic_payload(state: InterviewState, text: str) -> dict[str, Any] | None:
    stripped = " ".join((text or "").split())
    if not stripped:
        return {
            "response_type": "silence",
            "candidate_question_intent": None,
            "resume_skill_match": False,
        }
    if state.get("awaiting_think_confirmation") and YES_THINK_PATTERNS.search(stripped):
        return {
            "response_type": "think_request",
            "candidate_question_intent": None,
            "resume_skill_match": False,
        }
    if TEXT_SKIP_PATTERNS.search(stripped):
        skill = state.get("current_skill")
        return {
            "response_type": "clarification_question",
            "candidate_question_intent": "skip_question",
            "resume_skill_match": resume_mentions_skill(
                state.get("resume_parsed") or {},
                skill,
            ),
        }
    if TEXT_REPEAT_PATTERNS.search(stripped):
        return {
            "response_type": "clarification_question",
            "candidate_question_intent": "repeat_question",
            "resume_skill_match": False,
        }
    if TEXT_REPHRASE_PATTERNS.search(stripped):
        return {
            "response_type": "clarification_question",
            "candidate_question_intent": "rephrase_question",
            "resume_skill_match": False,
        }
    if TEXT_IRRELEVANT_PATTERNS.search(stripped):
        return {
            "response_type": "irrelevant_answer",
            "candidate_question_intent": None,
            "resume_skill_match": False,
        }
    return None


def classify_response(state: InterviewState) -> dict[str, Any]:
    """Synchronous heuristic classifier used by tests and deterministic routes."""

    event = state.get("normalized_candidate_event") or {}
    payload = _event_payload(state) or _heuristic_payload(
        state,
        str(event.get("text") or ""),
    )
    if payload is None:
        payload = {
            "response_type": "answer",
            "candidate_question_intent": None,
            "resume_skill_match": False,
        }
    return {
        "last_response_type": payload["response_type"],
        "last_classification": payload,
    }


def _classification_messages(state: InterviewState) -> list[dict[str, str]]:
    event = state.get("normalized_candidate_event") or {}
    skill = state.get("current_skill")
    context = {
        "candidate_text": event.get("text") or "",
        "current_question": state.get("current_question_text"),
        "current_section": state.get("current_section"),
        "current_skill": skill,
        "resume_skills": extract_resume_skills(state.get("resume_parsed") or {})[:30],
        "resume_mentions_current_skill_static_check": resume_mentions_skill(
            state.get("resume_parsed") or {},
            skill,
        ),
        "allowed_response_types": [
            "silence",
            "clarification_question",
            "irrelevant_answer",
            "answer",
        ],
        "allowed_candidate_question_intents": [
            "repeat_question",
            "rephrase_question",
            "skip_question",
            None,
        ],
    }
    return [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Return JSON matching this schema: "
                f"{compact_json(CandidateResponseClassification.model_json_schema(), max_chars=2600)}. "
                f"Context: {compact_json(context, max_chars=2400)}"
            ),
        },
    ]


async def classify_response_node(state: InterviewState) -> dict[str, Any]:
    event = state.get("normalized_candidate_event") or {}
    text = str(event.get("text") or "")

    payload = _event_payload(state) or _heuristic_payload(state, text)
    if payload is None:
        payload = model_to_dict(
            await classify_json(
                _classification_messages(state),
                CandidateResponseClassification,
            )
        )

    response_type = str(payload.get("response_type") or "answer")
    is_substantial = False
    substantiality_reason = response_type
    if response_type == "answer":
        is_substantial, substantiality_reason = is_substantial_answer(state, text)

    next_action = state.get("next_action")
    should_close = bool(state.get("should_close"))
    if response_type == "timer_expired":
        next_action = "complete"
        should_close = True

    logger.info(
        "classified candidate response",
        extra={
            "candidate_assessment_id": state.get("candidate_assessment_id"),
            "question_id": state.get("current_question_id"),
            "response_type": response_type,
            "is_substantial": is_substantial,
            "reason": substantiality_reason,
        },
    )

    return {
        "last_response_type": response_type,
        "last_response_substantial": is_substantial,
        "last_response_reason": substantiality_reason,
        "last_classification": payload,
        "next_action": next_action,
        "should_close": should_close,
    }
