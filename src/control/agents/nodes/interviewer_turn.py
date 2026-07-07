"""Two-stage candidate routing and live interviewer response handling.

High-confidence deterministic paths remain local. Other utterances first receive a
small classification-only call. A second model call runs only when a substantial
answer needs evaluation/question generation or a dynamic clarification is required.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.control.agents.key_routing import active_turn_key_slot
from src.control.agents.nodes.apply_question import apply_resolved_question
from src.control.agents.nodes.classify_response import (
    BEHAVIOURAL_ANSWER_WORD_THRESHOLD,
    BEHAVIOURAL_SUBSTANTIAL_WORD_THRESHOLD,
    _deterministic_classification,
    _self_intro_combined_text,
    _spoken_word_count,
    _turn_violations,
    is_deterministic_classification_source,
    self_intro_is_substantial,
    technical_answer_signal_is_present,
)
from src.control.agents.nodes.persist_turn import schedule_elapsed_persistence
from src.control.agents.nodes.question_diversity import (
    framing_hint,
    recent_acknowledgements,
    recent_question_stems,
    validate_generated_question,
)
from src.control.agents.nodes.question_strategy import (
    behavioural_questions,
    difficulty_plan,
    is_self_intro_phase,
    questions_for_skill,
    resume_has_skill,
    section_expected_signals,
)
from src.control.agents.nodes.time_manager import decide_time_action
from src.control.agents.prompts import (
    CLASSIFICATION_SYSTEM_PROMPT,
    LIVE_INTERVIEWER_SYSTEM_PROMPT,
)
from src.control.agents.state import InterviewState
from src.control.agents.templates import choose_template
from src.core.services import llm_service
from src.schemas.prompts import (
    CandidateResponseClassification,
    LiveInterviewerResponse,
)

logger = logging.getLogger(__name__)

_CONTEXT_HISTORY_LIMIT = 3


def _self_intro_state_updates(state: InterviewState, text: str) -> dict[str, Any]:
    """Persist cumulative self-intro speech across split turns and pauses."""

    if not is_self_intro_phase(state):
        return {}
    normalized = " ".join(str(text or "").split())
    if not normalized:
        return {}
    return {
        "self_intro_accumulated_response": _self_intro_combined_text(
            state,
            normalized,
        )
    }


def _should_finish_deterministically(
    det: dict[str, Any] | None,
) -> bool:
    """Route trusted static cases without a classifier call."""

    return det is not None and det.get("response_type") in {
        "silence",
        "clarification",
        "irrelevant",
    }


def _safe_classification_fallback(text: str) -> CandidateResponseClassification:
    """Prefer a non-destructive answer route if the classifier is unavailable."""

    return CandidateResponseClassification(
        response_type="answer",
        clarification_type=None,
        is_substantial=_spoken_word_count(text) > 2,
        interview_meta_type=None,
    )


def _guard_against_overstrict_irrelevance(
    state: InterviewState,
    text: str,
    classification: CandidateResponseClassification,
) -> CandidateResponseClassification:
    """Promote topical technical attempts that the small classifier rejected."""

    if (
        classification.response_type != "irrelevant"
        or not technical_answer_signal_is_present(state, text)
    ):
        return classification
    return CandidateResponseClassification(
        response_type="answer",
        clarification_type=None,
        is_substantial=_spoken_word_count(text) >= 10,
        interview_meta_type=None,
    )


def _safe_live_fallback(
    response_mode: str,
    *,
    is_technical_answer: bool,
) -> LiveInterviewerResponse | None:
    """Return a valid result that lets existing local fallbacks finish the turn."""

    if response_mode == "answer":
        return LiveInterviewerResponse(
            response_mode="answer",
            answer_strength="adequate" if is_technical_answer else None,
            acknowledgement=None,
            question_text=None,
            topic=None,
            clarification_response=None,
        )
    return None


def _resolve_target(
    state: InterviewState,
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the section the next question would belong to, given the time action."""

    if decision["action"] == "transition" and decision.get("next_section"):
        section = decision["next_section"]
        skill = str(section.get("skill") or section.get("section_name") or "").strip()
        return {
            "kind": str(section.get("section_kind") or "technical"),
            "skill": skill or None,
            "signals": list(section.get("expected_signals") or []),
            "entering_new_section": True,
        }
    return {
        "kind": str(state.get("current_section_kind") or "technical"),
        "skill": state.get("current_technical_skill"),
        "signals": section_expected_signals(state),
        "entering_new_section": False,
    }


def _difficulty_plan_for_target(
    state: InterviewState,
    target: dict[str, Any],
) -> dict[str, dict[str, Any]] | None:
    """Return a technical plan for the resolved target, if applicable."""

    if target["kind"] != "technical":
        return None
    return difficulty_plan(
        state,
        skill=str(target["skill"] or state.get("current_technical_skill") or "skill"),
        entering_new_section=bool(target["entering_new_section"]),
    )


def _classification_messages(state: InterviewState) -> list[dict[str, str]]:
    """Build the minimal classification-only request."""

    context = {
        "current_question": state.get("current_question_text") or "",
        "candidate_response": state.get("previous_candidate_response") or "",
        "section_kind": state.get("current_section_kind") or "technical",
    }
    return [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, default=str),
        },
    ]


def _response_mode(
    classification: CandidateResponseClassification,
) -> str | None:
    """Map classification to the only stage-two modes that need model output."""

    if classification.response_type == "answer" and classification.is_substantial:
        return "answer"
    if (
        classification.response_type == "clarification"
        and classification.clarification_type
        in {
            "question_doubt",
            "rephrase_question",
        }
    ):
        return classification.clarification_type
    return None


def _live_interviewer_messages(
    state: InterviewState,
    *,
    classification: CandidateResponseClassification,
    target: dict[str, Any],
    decision: dict[str, Any],
    ask_next_question: bool,
    must_close: bool,
    plan: dict[str, dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Assemble stage-two evaluation/question or clarification context."""

    response_mode = _response_mode(classification)
    if response_mode in {"question_doubt", "rephrase_question"}:
        clarification_context = {
            "response_mode": response_mode,
            "current_question": state.get("current_question_text") or "",
            "candidate_response": state.get("previous_candidate_response") or "",
            "section_kind": state.get("current_section_kind") or "technical",
            "current_technical_skill": state.get("current_technical_skill"),
            "question_difficulty": state.get("current_question_difficulty"),
        }
        return [
            {"role": "system", "content": LIVE_INTERVIEWER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    clarification_context,
                    ensure_ascii=False,
                    default=str,
                ),
            },
        ]

    is_behavioural = target["kind"] == "behavioural_cultural"
    transition = None
    if decision["action"] == "transition":
        transition = {
            "from": decision.get("current_label"),
            "to": decision.get("next_label"),
        }

    follow_interesting_thread = bool(
        (plan or {}).get("strong", {}).get("follow_interesting_thread")
    )

    sequence_number = len(state.get("asked_questions") or []) + 1
    seed = str(state.get("question_variation_seed") or "")
    context: dict[str, Any] = {
        "response_mode": "answer",
        "section_kind": target["kind"],
        "answer_section_kind": state.get("current_section_kind") or "technical",
        "next_section_kind": target["kind"],
        "previous_candidate_response": state.get("previous_candidate_response") or "",
        "current_question": state.get("current_question_text") or "",
        "ask_next_question": ask_next_question,
        "must_close": must_close,
        "transition": transition,
        "follow_interesting_thread": follow_interesting_thread,
        "inferred_difficulty": state.get("inferred_difficulty"),
        "question_framing_hint": framing_hint(seed, sequence_number),
        "recent_question_stems": recent_question_stems(state),
        "recent_acknowledgements": recent_acknowledgements(
            state,
            limit=_CONTEXT_HISTORY_LIMIT,
        ),
    }

    if is_behavioural:
        asked = behavioural_questions(state)
        context.update(
            {
                "expected_signals": target["signals"],
                "questions_already_asked": [
                    item.get("question_text") for item in asked
                ][-_CONTEXT_HISTORY_LIMIT:],
                "signals_already_used": [
                    item.get("topic") for item in asked if item.get("topic")
                ][-_CONTEXT_HISTORY_LIMIT:],
            }
        )
    else:
        skill = str(
            target["skill"]
            or state.get("current_technical_skill")
            or "the role's core skill"
        )
        asked = questions_for_skill(state, skill)
        context.update(
            {
                "current_technical_skill": skill,
                "difficulty_plan": plan or {},
                "questions_already_asked_for_skill": [
                    item.get("question_text") for item in asked
                ][-_CONTEXT_HISTORY_LIMIT:],
                "topics_already_used_for_skill": list(
                    (state.get("used_topics_by_skill") or {}).get(
                        skill.casefold(),
                        [],
                    )
                )[-_CONTEXT_HISTORY_LIMIT:],
                "resume_context": state.get("resume_context") or {},
            }
        )

    return [
        {"role": "system", "content": LIVE_INTERVIEWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, default=str),
        },
    ]


def _classification_updates(
    state: InterviewState,
    *,
    response_type: str,
    clarification_type: str | None,
    is_substantial: bool | None,
    source: str,
    interview_meta_type: str | None = None,
    question_doubt_response: str | None = None,
    latest_evaluation: dict[str, Any] | None = None,
    evaluation_source: str | None = None,
) -> dict[str, Any]:
    """Build shared classification, violation, and candidate-turn updates."""

    classification = {
        "response_type": response_type,
        "clarification_type": clarification_type,
        "is_substantial": is_substantial,
        "interview_meta_type": interview_meta_type,
        "question_doubt_response": question_doubt_response,
    }
    resume_skill_match = clarification_type == "skip_question" and resume_has_skill(
        state,
        state.get("current_technical_skill"),
    )
    new_violations = _turn_violations(
        state,
        response_type=response_type,
        clarification_type=clarification_type,
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
    if latest_evaluation is not None:
        metadata.update(
            {
                "live_evaluation": latest_evaluation,
                "evaluation_source": evaluation_source or source,
            }
        )
    pending_candidate_turn.update(
        {
            "response_type": response_type,
            "metadata": metadata,
        }
    )
    return {
        "last_classification": classification,
        "classification_source": source,
        "last_response_type": response_type,
        "last_response_substantial": is_substantial,
        "last_skip_resume_skill_match": resume_skill_match,
        "latest_evaluation": latest_evaluation,
        "pending_candidate_turn": pending_candidate_turn,
        "violations_to_persist": new_violations,
        "recent_violations": recent_violations,
    }


def _finish_deterministic(
    state: InterviewState,
    det: dict[str, Any],
    started_at: float,
) -> dict[str, Any]:
    """Handle a trusted silence/clarification entirely from templates (no LLM)."""

    response_type = str(det["response_type"])
    clarification_value = det.get("clarification_type")
    clarification_type = str(clarification_value) if clarification_value else None
    is_substantial = det.get("is_substantial")
    common_updates = _classification_updates(
        state,
        response_type=response_type,
        clarification_type=clarification_type,
        is_substantial=is_substantial,
        source="deterministic",
    )

    silence_stage = state.get("silence_stage") or "none"
    if response_type != "silence" and clarification_type not in {
        "time_to_think",
        "decline_think_time",
    }:
        silence_stage = "none"

    logger.info(
        "Interviewer turn resolved deterministically",
        extra={
            "candidate_assessment_id": state.get("candidate_assessment_id"),
            "response_type": response_type,
            "clarification_type": clarification_type,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
        },
    )
    return {
        **common_updates,
        "pregenerated_closing_lead": None,
        "pending_clarification_text": None,
        "silence_stage": silence_stage,
        "next_action": "generate_bot_response",
    }


def _finish_classification_only(
    state: InterviewState,
    classification: CandidateResponseClassification,
    source: str,
    started_at: float,
) -> dict[str, Any]:
    """Commit a classification whose response is fully handled by templates."""

    response_type = classification.response_type
    clarification_type = classification.clarification_type
    common_updates = _classification_updates(
        state,
        response_type=response_type,
        clarification_type=clarification_type,
        is_substantial=classification.is_substantial,
        interview_meta_type=classification.interview_meta_type,
        source=source,
    )
    silence_stage = state.get("silence_stage") or "none"
    if clarification_type not in {"time_to_think", "decline_think_time"}:
        silence_stage = "none"

    logger.info(
        "Candidate response resolved after classification",
        extra={
            "candidate_assessment_id": state.get("candidate_assessment_id"),
            "response_type": response_type,
            "clarification_type": clarification_type,
            "interview_meta_type": classification.interview_meta_type,
            "classification_source": source,
            "stage_two_skipped": True,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
        },
    )
    return {
        **common_updates,
        "pregenerated_closing_lead": None,
        "pending_clarification_text": None,
        "silence_stage": silence_stage,
        "next_action": "generate_bot_response",
    }


def _apply_result(
    state: InterviewState,
    classification: CandidateResponseClassification,
    result: LiveInterviewerResponse | None,
    decision: dict[str, Any],
    target: dict[str, Any],
    plan: dict[str, dict[str, Any]] | None,
    classification_source: str,
    response_source: str | None,
    started_at: float,
) -> dict[str, Any]:
    """Translate classified stage-two output into deterministic graph updates."""

    text = str(state.get("previous_candidate_response") or "")
    response_type = classification.response_type
    clarification_type = classification.clarification_type
    is_substantial = classification.is_substantial
    answer_strength = result.answer_strength if result is not None else None

    if response_type == "answer" and is_self_intro_phase(state):
        # The self-introduction elaboration rule uses cumulative spoken words.
        is_substantial = self_intro_is_substantial(state, text)
    elif (
        response_type == "answer"
        and state.get("current_section_kind") == "behavioural_cultural"
        and is_deterministic_classification_source(classification_source)
    ):
        word_count = _spoken_word_count(text)
        if word_count > BEHAVIOURAL_ANSWER_WORD_THRESHOLD:
            is_substantial = word_count > BEHAVIOURAL_SUBSTANTIAL_WORD_THRESHOLD

    latest_evaluation: dict[str, Any] | None = None
    skill_streaks = {
        key: dict(value)
        for key, value in (state.get("skill_evaluation_streaks") or {}).items()
    }
    if (
        response_type == "answer"
        and is_substantial
        and state.get("current_section_kind") == "technical"
        and answer_strength
    ):
        latest_evaluation = {"strength": answer_strength}
        if state.get("current_technical_skill"):
            key = str(state["current_technical_skill"]).casefold()
            current = dict(
                skill_streaks.get(key) or {"weak": 0, "adequate": 0, "strong": 0}
            )
            for strength in ("weak", "adequate", "strong"):
                current[strength] = (
                    int(current.get(strength) or 0) + 1
                    if strength == answer_strength
                    else 0
                )
            skill_streaks[key] = current

    base: dict[str, Any] = {
        **_classification_updates(
            state,
            response_type=response_type,
            clarification_type=clarification_type,
            is_substantial=is_substantial,
            interview_meta_type=classification.interview_meta_type,
            source=classification_source,
            question_doubt_response=(
                result.clarification_response
                if clarification_type == "question_doubt" and result is not None
                else None
            ),
            latest_evaluation=latest_evaluation,
            evaluation_source=response_source,
        ),
        "skill_evaluation_streaks": skill_streaks,
        "pregenerated_closing_lead": None,
        "pending_clarification_text": None,
        **_self_intro_state_updates(state, text),
    }

    logger.info(
        "Interviewer turn resolved after staged processing",
        extra={
            "candidate_assessment_id": state.get("candidate_assessment_id"),
            "response_type": response_type,
            "is_substantial": is_substantial,
            "answer_strength": answer_strength,
            "time_action": decision["action"],
            "classification_source": classification_source,
            "response_source": response_source,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
        },
    )

    forced_behavioural_preface = None
    if decision.get("transition_reason") == "behavioural_time_rescue":
        forced_behavioural_preface = choose_template("behavioural_forced_transition")

    intro_transition_preface = None
    if (
        decision.get("action") == "transition"
        and decision.get("transition_reason") == "self_introduction_complete"
    ):
        intro_ack = choose_template("self_intro_completion_ack")
        section_bridge = choose_template(
            "section_transition",
            current_section=decision.get("current_label") or "your introduction",
            next_section=decision.get("next_label") or "the next section",
        )
        intro_transition_preface = f"{intro_ack} {section_bridge}".strip()

    if response_type == "answer" and is_substantial:
        schedule_elapsed_persistence(
            state,
            elapsed_secs=int(decision["timing_updates"]["elapsed_secs"]),
        )

        if decision["action"] == "close":
            lead = str(result.acknowledgement or "").strip() if result else ""
            return {
                **base,
                **decision["timing_updates"],
                "pregenerated_closing_lead": lead or None,
                "should_close": True,
                "closing_reason": decision.get("closing_reason"),
                "pending_section_index": None,
                "response_preface_text": None,
                "silence_stage": "none",
                "next_action": "generate_closing",
            }

        question_text = str(result.question_text or "").strip() if result else ""
        resolved_question: dict[str, Any] | None = None
        if question_text:
            acknowledgement = str(result.acknowledgement or "").strip()
            if forced_behavioural_preface:
                acknowledgement = forced_behavioural_preface
            elif response_source is None:
                acknowledgement = ""
            topic = str(result.topic or "").strip()
            if target["kind"] == "technical":
                strength_key = (
                    answer_strength
                    if (plan and answer_strength in plan)
                    else "adequate"
                )
                chosen = (plan or {}).get(strength_key) or {
                    "difficulty": None,
                    "probe_deeper": False,
                    "follow_interesting_thread": False,
                }
                follow_thread = bool(chosen.get("follow_interesting_thread"))
                resolved_question = {
                    "section_kind": "technical",
                    "skill": target["skill"],
                    "question_text": question_text,
                    "acknowledgement": acknowledgement,
                    "topic": topic or f"{target['skill'] or 'core'} fundamentals",
                    "difficulty": chosen.get("difficulty"),
                    "probe_deeper": bool(chosen.get("probe_deeper")),
                    "follow_interesting_thread": follow_thread,
                }
                skill_name = str(target["skill"] or "")
                try:
                    validate_generated_question(
                        question_text=question_text,
                        topic=str(resolved_question["topic"]),
                        skill=skill_name,
                        asked_questions=questions_for_skill(state, skill_name),
                        used_topics=list(
                            (state.get("used_topics_by_skill") or {}).get(
                                skill_name.casefold(),
                                [],
                            )
                        ),
                        allow_related_probe=bool(resolved_question["probe_deeper"]),
                    )
                except ValueError as exc:
                    logger.info(
                        "Pregenerated technical question rejected: %s",
                        exc,
                    )
                    resolved_question = None
            else:
                asked_behavioural = behavioural_questions(state)
                resolved_question = {
                    "section_kind": "behavioural_cultural",
                    "skill": None,
                    "question_text": question_text,
                    "acknowledgement": acknowledgement,
                    "topic": topic or "behavioural signal",
                    "difficulty": None,
                    "probe_deeper": False,
                }
                try:
                    validate_generated_question(
                        question_text=question_text,
                        topic=str(resolved_question["topic"]),
                        skill="",
                        asked_questions=asked_behavioural,
                        used_topics=[
                            str(item.get("topic") or "")
                            for item in asked_behavioural
                            if item.get("topic")
                        ],
                        allow_related_probe=False,
                    )
                except ValueError as exc:
                    logger.info(
                        "Pregenerated behavioural question rejected: %s",
                        exc,
                    )
                    resolved_question = None

        if resolved_question:
            merged_state: InterviewState = {
                **state,
                **decision["timing_updates"],
                "pending_section_index": decision["pending_section_index"],
                "suppress_previous_context_for_next_question": decision[
                    "suppress_previous_context_for_next_question"
                ],
                "transition_reason": decision.get("transition_reason"),
            }
            applied = apply_resolved_question(
                merged_state,
                acknowledgement=str(
                    resolved_question.get("acknowledgement") or ""
                ).strip(),
                question_text=str(resolved_question["question_text"]).strip(),
                topic=str(resolved_question.get("topic") or "").strip(),
                current_skill=(
                    str(resolved_question.get("skill") or "") or None
                    if target["kind"] == "technical"
                    else None
                ),
                current_difficulty=resolved_question.get("difficulty"),
                probe_deeper=bool(resolved_question.get("probe_deeper")),
                follow_interesting_thread=bool(
                    resolved_question.get("follow_interesting_thread")
                ),
            )
            return {
                **base,
                **decision["timing_updates"],
                **applied,
                "should_close": False,
                "closing_reason": None,
                "barge_in_triggered": False,
                "silence_stage": "none",
            }

        return {
            **base,
            **decision["timing_updates"],
            "pending_section_index": decision["pending_section_index"],
            "suppress_previous_context_for_next_question": decision[
                "suppress_previous_context_for_next_question"
            ],
            "transition_reason": decision.get("transition_reason"),
            "response_preface_text": forced_behavioural_preface
            or intro_transition_preface,
            "should_advance_question": True,
            "should_close": False,
            "closing_reason": None,
            "barge_in_triggered": False,
            "silence_stage": "none",
            "next_action": "generate_next_question",
        }

    pending_clarification_text = None
    if response_type == "clarification" and clarification_type in {
        "rephrase_question",
        "question_doubt",
    }:
        pending_clarification_text = (
            str(result.clarification_response or "").strip() or None
            if result is not None
            else None
        )

    silence_stage = state.get("silence_stage") or "none"
    if clarification_type != "time_to_think":
        silence_stage = "none"

    return {
        **base,
        "pending_clarification_text": pending_clarification_text,
        "silence_stage": silence_stage,
        "next_action": "generate_bot_response",
    }


def _validate_live_response(
    classification: CandidateResponseClassification,
    state: InterviewState,
    result: LiveInterviewerResponse,
) -> None:
    """Reject a stage-two response that contradicts its fixed routing mode."""

    expected_mode = _response_mode(classification)
    if result.response_mode != expected_mode:
        raise ValueError(
            f"stage-two mode {result.response_mode!r} does not match "
            f"classification mode {expected_mode!r}"
        )
    if expected_mode != "answer":
        return
    is_technical = state.get("current_section_kind") == "technical"
    if is_technical and result.answer_strength is None:
        raise ValueError("technical answer requires answer_strength")
    if not is_technical and result.answer_strength is not None:
        raise ValueError("non-technical answer requires answer_strength=null")


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
