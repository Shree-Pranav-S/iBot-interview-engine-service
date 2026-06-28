"""Deterministic fast paths and strict LLM response classification."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from src.control.agents.nodes.llm_helpers import classify_with_schema
from src.control.agents.prompts import CLASSIFICATION_SYSTEM_PROMPT
from src.control.agents.state import InterviewState
from src.schemas.prompts import CandidateResponseClassification
from src.utils.interview_graph import deterministic_violation_id, utc_now_iso

logger = logging.getLogger(__name__)

REPEAT_PATTERN = re.compile(
    r"\b(repeat (?:it|that|this question|that question|the question)|"
    r"say (?:it|that) again|hear (?:it|that|this question|the question) again|"
    r"once more)\b",
    re.IGNORECASE,
)
REPHRASE_PATTERN = re.compile(
    r"\b((?:can|could|would|will) you (?:please )?rephrase|please rephrase|"
    r"rephrase (?:it|that|this|the question)|"
    r"phrase (?:it|that) differently|put (?:it|that) another way|"
    r"simplify (?:it|that|the question)|make the question (?:simpler|clearer))\b",
    re.IGNORECASE,
)
SKIP_PATTERN = re.compile(
    r"\b(skip (?:it|this|that|the question)|pass (?:on )?(?:it|this|that)|"
    r"move (?:on|to the next question)|next question|"
    r"i (?:do not|don't|dont) know(?: the answer)?|"
    r"i (?:cannot|can't|cant) answer|no idea)\b",
    re.IGNORECASE,
)
THINK_PATTERN = re.compile(
    r"\b(?:i\s+)?(?:need|want|would like|could use)\s+"
    r"(?:some|a|a few|more|a little|a bit of)?\s*"
    r"(?:time|a moment|seconds?|a minute)\s+(?:to\s+)?"
    r"(?:think|consider|prepare|gather my thoughts)\b|"
    r"\b(?:can|could|may)\s+i\s+(?:take|have|get)\s+"
    r"(?:a|some|a few)?\s*(?:moment|time|seconds?|minute)\s+"
    r"(?:to\s+)?(?:think|consider|prepare)\b|"
    r"\b(?:give me|can i have|could i have|may i have)\s+"
    r"(?:a|one|some)?\s*(?:moment|minute|few seconds)\b",
    re.IGNORECASE,
)
YES_PATTERN = re.compile(
    r"^\s*(yes|yeah|yep|sure|please|okay|ok|i do|that would help)\b",
    re.IGNORECASE,
)
NO_PATTERN = re.compile(
    r"^\s*(no|nope|not really|i am (?:fine|ready)|i'm (?:fine|ready))\b",
    re.IGNORECASE,
)
YEARS_OF_EXPERIENCE_PATTERN = re.compile(
    r"\b(?:i\s+(?:have|bring|possess))\s+"
    r"(?:about|around|roughly|nearly|over|more than)?\s*"
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b"
    r"(?:\s+of\s+(?:professional\s+|work\s+|industry\s+)?experience)?|"
    r"\bwith\s+(?:about|around|roughly|nearly|over|more than)?\s*"
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+of\s+"
    r"(?:professional\s+|work\s+|industry\s+)?experience\b",
    re.IGNORECASE,
)
SELF_INTRO_PROFESSIONAL_PATTERN = re.compile(
    r"\b(?:background|career|education|experience|professional|"
    r"work(?:ed|ing)?|roles?|developers?|engineers?|projects?|skills?|"
    r"technology|technologies|frameworks?|responsibilit(?:y|ies)|"
    r"proficien(?:t|cy)|speciali[sz](?:e|ed|ation)|currently)\b",
    re.IGNORECASE,
)


def _result(
    *,
    response_type: str,
    clarification_type: str | None,
    is_substantial: bool | None,
    reason: str,
    question_doubt_response: str | None = None,
) -> dict[str, Any]:
    return {
        "response_type": response_type,
        "clarification_type": clarification_type,
        "is_substantial": is_substantial,
        "question_doubt_response": question_doubt_response,
        "reason": reason,
    }


def _deterministic_classification(
    state: InterviewState,
    text: str,
) -> dict[str, Any] | None:
    event = state.get("candidate_event") or {}
    if isinstance(event, dict) and event.get("event_type") == "silence_timeout":
        return _result(
            response_type="silence",
            clarification_type=None,
            is_substantial=None,
            reason="LiveKit reported five seconds without candidate speech.",
        )

    if state.get("silence_stage") == "awaiting_think_confirmation":
        if YES_PATTERN.search(text):
            return _result(
                response_type="clarification",
                clarification_type="time_to_think",
                is_substantial=None,
                reason="Candidate accepted the offered thinking time.",
            )
        if NO_PATTERN.search(text):
            return _result(
                response_type="clarification",
                clarification_type="decline_think_time",
                is_substantial=None,
                reason="Candidate declined the offered thinking time.",
            )

    if SKIP_PATTERN.search(text):
        return _result(
            response_type="clarification",
            clarification_type="skip_question",
            is_substantial=None,
            reason="High-confidence skip or cannot-answer phrase.",
        )
    if REPEAT_PATTERN.search(text):
        return _result(
            response_type="clarification",
            clarification_type="repeat_question",
            is_substantial=None,
            reason="High-confidence request to repeat the question.",
        )
    if REPHRASE_PATTERN.search(text):
        return _result(
            response_type="clarification",
            clarification_type="rephrase_question",
            is_substantial=None,
            reason="High-confidence request to rephrase the question.",
        )
    if THINK_PATTERN.search(text):
        return _result(
            response_type="clarification",
            clarification_type="time_to_think",
            is_substantial=None,
            reason="High-confidence request for thinking time.",
        )

    if bool(state.get("is_self_introduction")):
        words = re.findall(r"\b[\w+#.-]+\b", text)
        has_professional_detail = bool(SELF_INTRO_PROFESSIONAL_PATTERN.search(text))
        if len(words) >= 18 or (len(words) >= 6 and has_professional_detail):
            return _result(
                response_type="answer",
                clarification_type=None,
                is_substantial=len(words) >= 12,
                reason=(
                    "Deterministic self-introduction safeguard recognized "
                    "professional background or sufficiently detailed speech."
                ),
            )
    return None


def _classification_messages(state: InterviewState) -> list[dict[str, str]]:
    duration_ms = state.get("previous_response_duration_ms")
    context = {
        "previous_candidate_response": state.get("previous_candidate_response") or "",
        "previous_question": state.get("current_question_text") or "",
        "is_self_introduction": bool(state.get("is_self_introduction")),
        "resume_context": state.get("resume_context")
        or {
            "skills": [],
            "experience_years": 0,
        },
        "response_duration_seconds": (
            round(int(duration_ms) / 1000, 2) if duration_ms is not None else None
        ),
        "schema": CandidateResponseClassification.model_json_schema(),
    }
    return [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, default=str),
        },
    ]


def _safe_fallback(text: str) -> CandidateResponseClassification:
    """Keep a live interview moving without accepting malformed model JSON."""

    words = re.findall(r"\b[\w+#.-]+\b", text)
    return CandidateResponseClassification(
        response_type="answer",
        clarification_type=None,
        is_substantial=len(words) > 2,
        question_doubt_response=None,
        reason="Conservative answer fallback after classification service failure.",
    )


def _resume_has_skill(state: InterviewState) -> bool:
    skill = str(state.get("current_technical_skill") or "").strip().casefold()
    if not skill:
        return False
    for raw in (state.get("resume_context") or {}).get("skills", []):
        resume_skill = str(raw).strip().casefold()
        if resume_skill == skill:
            return True
        if (
            resume_skill
            and min(len(resume_skill), len(skill)) >= 3
            and (resume_skill in skill or skill in resume_skill)
        ):
            return True
    return False


def _violation(
    state: InterviewState,
    violation_type: str,
    *,
    severity: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "violation_id": deterministic_violation_id(
            str(state["interview_session_id"]),
            int(state.get("turn_number") or 1),
            violation_type,
        ),
        "turn_number": int(state.get("turn_number") or 1),
        "violation_type": violation_type,
        "candidate_transcript": str(state.get("previous_candidate_response") or ""),
        "severity": severity,
        "timestamp": utc_now_iso(),
        "metadata": metadata,
    }


def _turn_violations(
    state: InterviewState,
    *,
    response_type: str,
    clarification_type: str | None,
    resume_skill_match: bool,
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    if response_type == "irrelevant":
        violations.append(
            _violation(
                state,
                "irrelevant_response",
                severity="low",
                metadata={
                    "question_id": state.get("current_question_id"),
                    "current_skill": state.get("current_technical_skill"),
                },
            )
        )

    if clarification_type == "skip_question" and resume_skill_match:
        violations.append(
            _violation(
                state,
                "skipped_resume_skill",
                severity="medium",
                metadata={
                    "question_id": state.get("current_question_id"),
                    "current_skill": state.get("current_technical_skill"),
                },
            )
        )

    if response_type == "answer":
        match = YEARS_OF_EXPERIENCE_PATTERN.search(
            str(state.get("previous_candidate_response") or "")
        )
        resume_years = float(
            (state.get("resume_context") or {}).get("experience_years") or 0
        )
        if match and resume_years > 0:
            claimed_years = float(match.group(1) or match.group(2))
            # Allow normal rounding of a fractional parsed estimate, but flag a
            # materially different whole-year claim.
            if abs(claimed_years - resume_years) > 0.5:
                violations.append(
                    _violation(
                        state,
                        "experience_years_mismatch",
                        severity="medium",
                        metadata={
                            "resume_experience_years": resume_years,
                            "claimed_experience_years": claimed_years,
                        },
                    )
                )
    return violations


async def classify_candidate_response(state: InterviewState) -> dict[str, Any]:
    """Classify speech, bypassing the LLM for deterministic high-confidence cases."""

    text = str(state.get("previous_candidate_response") or "")
    classification = _deterministic_classification(state, text)
    source = "deterministic"

    if classification is None:
        try:
            model_result = await classify_with_schema(
                _classification_messages(state),
                CandidateResponseClassification,
            )
        except Exception:
            logger.exception(
                "Strict candidate-response classification failed",
                extra={
                    "candidate_assessment_id": state.get("candidate_assessment_id"),
                    "question_id": state.get("current_question_id"),
                },
            )
            model_result = _safe_fallback(text)
            source = "validated_fallback"
        else:
            source = "llm"
        classification = model_result.model_dump()

    response_type = str(classification["response_type"])
    is_substantial = classification.get("is_substantial")
    duration_ms = state.get("previous_response_duration_ms")
    self_intro_short_override = (
        response_type == "answer"
        and bool(state.get("is_self_introduction"))
        and not bool(state.get("self_intro_elaboration_requested"))
        and duration_ms is not None
        and int(duration_ms) <= 15_000
    )
    if self_intro_short_override:
        is_substantial = False
        classification["is_substantial"] = False
        classification["reason"] = (
            f"{classification['reason']} The first self-introduction response "
            "was no longer than 15 seconds, so phase-one policy requires elaboration."
        )[:240]

    clarification_type = classification.get("clarification_type")
    resume_skill_match = clarification_type == "skip_question" and _resume_has_skill(
        state
    )
    new_violations = _turn_violations(
        state,
        response_type=response_type,
        clarification_type=(str(clarification_type) if clarification_type else None),
        resume_skill_match=resume_skill_match,
    )
    recent_violations = [
        *list(state.get("recent_violations") or []),
        *new_violations,
    ][-20:]

    pending_candidate_turn = dict(state.get("pending_candidate_turn") or {})
    metadata = dict(pending_candidate_turn.get("metadata") or {})
    metadata.update(
        {
            "classification": classification,
            "classification_source": source,
            "clarification_type": clarification_type,
            "is_substantial": is_substantial,
        }
    )
    pending_candidate_turn.update(
        {
            "response_type": response_type,
            "metadata": metadata,
        }
    )

    silence_stage = state.get("silence_stage") or "none"
    if response_type != "silence" and classification.get("clarification_type") not in {
        "time_to_think",
        "decline_think_time",
    }:
        silence_stage = "none"

    return {
        "last_classification": classification,
        "classification_source": source,
        "last_response_type": response_type,
        "last_response_substantial": is_substantial,
        "last_skip_resume_skill_match": resume_skill_match,
        "pending_candidate_turn": pending_candidate_turn,
        "violations_to_persist": new_violations,
        "recent_violations": recent_violations,
        "silence_stage": silence_stage,
        "next_action": (
            "evaluate_answer"
            if (
                response_type == "answer"
                and is_substantial
                and state.get("current_section_kind") != "behavioural_cultural"
            )
            else (
                "check_time"
                if response_type == "answer" and is_substantial
                else "generate_bot_response"
            )
        ),
    }
