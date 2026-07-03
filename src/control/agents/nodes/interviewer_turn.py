"""Single merged interviewer turn: classify, evaluate, and respond in one call.

This node replaces the former three sequential LLM calls (classify -> evaluate ->
generate). For an unambiguous clarification or silence it still answers instantly
from templates with no model call. For everything else it makes ONE structured
``InterviewerTurnResponse`` call that routes the utterance, judges the answer, and
produces the next spoken question or clarification. The deterministic difficulty
plan and section-timing decision are computed before the call so adaptive control
stays fully deterministic while latency drops to a single round trip.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from src.config.settings import settings
from src.control.agents.key_routing import active_turn_key_slot
from src.control.agents.nodes.apply_question import apply_resolved_question
from src.control.agents.nodes.classify_response import (
    BEHAVIOURAL_ANSWER_WORD_THRESHOLD,
    BEHAVIOURAL_SUBSTANTIAL_WORD_THRESHOLD,
    _deterministic_classification,
    _resume_has_skill,
    _self_intro_combined_text,
    _spoken_word_count,
    _turn_violations,
    is_deterministic_classification_source,
    self_intro_is_substantial,
)
from src.control.agents.nodes.llm_helpers import interviewer_turn_with_schema
from src.control.agents.nodes.persist_turn import schedule_elapsed_persistence
from src.control.agents.nodes.question_diversity import (
    framing_hint,
    recent_question_stems,
    validate_generated_question,
)
from src.control.agents.nodes.question_strategy import (
    behavioural_questions,
    difficulty_plan,
    is_self_intro_phase,
    questions_for_skill,
    section_expected_signals,
)
from src.control.agents.nodes.time_manager import decide_time_action
from src.control.agents.prompts import INTERVIEWER_TURN_SYSTEM_PROMPT
from src.control.agents.state import InterviewState
from src.control.agents.templates import choose_template
from src.schemas.prompts import InterviewerTurnResponse

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


def _should_finish_deterministically(det: dict[str, Any] | None) -> bool:
    """Route silence, irrelevant, and regex-matched clarifications without an LLM call."""

    if det is None:
        return False
    return det.get("response_type") in ("silence", "irrelevant", "clarification")


def _safe_merged_fallback(
    state: InterviewState,
    text: str,
    det: dict[str, Any] | None,
) -> InterviewerTurnResponse:
    """Keep the interview on the current question when the merged LLM call fails."""

    if det is not None:
        response_type = str(det.get("response_type") or "")
        if response_type == "irrelevant":
            return InterviewerTurnResponse(
                response_type="irrelevant",
                clarification_type=None,
                is_substantial=None,
                answer_strength=None,
                acknowledgement=None,
                question_text=None,
                topic=None,
                clarification_response=None,
            )
        if response_type == "clarification":
            return InterviewerTurnResponse(
                response_type="clarification",
                clarification_type=det.get("clarification_type"),
                is_substantial=None,
                answer_strength=None,
                acknowledgement=None,
                question_text=None,
                topic=None,
                clarification_response=None,
            )
        if response_type == "answer" and det.get("is_substantial") is False:
            return InterviewerTurnResponse(
                response_type="answer",
                clarification_type=None,
                is_substantial=False,
                answer_strength=None,
                acknowledgement=None,
                question_text=None,
                topic=None,
                clarification_response=None,
            )

    return InterviewerTurnResponse(
        response_type="answer",
        clarification_type=None,
        is_substantial=False,
        answer_strength=None,
        acknowledgement=None,
        question_text=None,
        topic=None,
        clarification_response=None,
    )


def _recent_acknowledgements(state: InterviewState) -> list[str]:
    return [
        str(item.get("acknowledgement") or "").strip()
        for item in (state.get("asked_questions") or [])
        if str(item.get("acknowledgement") or "").strip()
    ][-_CONTEXT_HISTORY_LIMIT:]


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


def _interviewer_messages(
    state: InterviewState,
    *,
    target: dict[str, Any],
    decision: dict[str, Any],
    ask_next_question: bool,
    must_close: bool,
    plan: dict[str, dict[str, Any]] | None,
) -> list[dict[str, str]]:
    """Assemble the single merged classify/evaluate/generate prompt context."""

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
    total_mins = max(1, int(state.get("total_duration_secs") or 60) // 60)
    elapsed_mins = max(0, int(state.get("elapsed_secs") or 0) // 60)
    context: dict[str, Any] = {
        "section_kind": target["kind"],
        "previous_candidate_response": state.get("previous_candidate_response") or "",
        "current_question": state.get("current_question_text") or "",
        "is_self_introduction": is_self_intro_phase(state),
        "ask_next_question": ask_next_question,
        "must_close": must_close,
        "is_section_start": bool(target["entering_new_section"]),
        "transition": transition,
        "follow_interesting_thread": follow_interesting_thread,
        "inferred_difficulty": state.get("inferred_difficulty"),
        "elapsed_minutes": elapsed_mins,
        "total_minutes": total_mins,
        "question_variation_seed": state.get("question_variation_seed"),
        "question_sequence_number": sequence_number,
        "question_framing_hint": framing_hint(seed, sequence_number),
        "recent_question_stems": recent_question_stems(state),
        "recent_acknowledgements": _recent_acknowledgements(state),
        "filler_already_spoken": bool(settings.ENABLE_ACK_FILLER),
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
        {"role": "system", "content": INTERVIEWER_TURN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, default=str),
        },
    ]


def _finish_deterministic(
    state: InterviewState,
    det: dict[str, Any],
    started_at: float,
) -> dict[str, Any]:
    """Handle a trusted silence/clarification entirely from templates (no LLM)."""

    response_type = str(det["response_type"])
    clarification_type = det.get("clarification_type")
    is_substantial = det.get("is_substantial")
    classification = {
        "response_type": response_type,
        "clarification_type": clarification_type,
        "is_substantial": is_substantial,
        "question_doubt_response": None,
    }
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
            "classification_source": "deterministic",
            "clarification_type": clarification_type,
            "is_substantial": is_substantial,
        }
    )
    pending_candidate_turn.update(
        {"response_type": response_type, "metadata": metadata}
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
        "last_classification": classification,
        "classification_source": "deterministic",
        "last_response_type": response_type,
        "last_response_substantial": is_substantial,
        "last_skip_resume_skill_match": resume_skill_match,
        "latest_evaluation": None,
        "pending_candidate_turn": pending_candidate_turn,
        "violations_to_persist": new_violations,
        "recent_violations": recent_violations,
        "pregenerated_question": None,
        "pregenerated_closing_lead": None,
        "pending_clarification_text": None,
        "silence_stage": silence_stage,
        "next_action": "generate_bot_response",
    }


def _apply_result(
    state: InterviewState,
    result: InterviewerTurnResponse,
    decision: dict[str, Any],
    target: dict[str, Any],
    plan: dict[str, dict[str, Any]] | None,
    source: str,
    started_at: float,
) -> dict[str, Any]:
    """Translate one merged model response into deterministic graph state updates."""

    text = str(state.get("previous_candidate_response") or "")
    response_type = result.response_type
    clarification_type = result.clarification_type
    is_substantial = result.is_substantial

    if response_type == "answer" and is_self_intro_phase(state):
        # The self-introduction elaboration rule uses cumulative spoken words.
        is_substantial = self_intro_is_substantial(state, text)
    elif (
        response_type == "answer"
        and state.get("current_section_kind") == "behavioural_cultural"
        and is_deterministic_classification_source(source)
    ):
        word_count = _spoken_word_count(text)
        if word_count > BEHAVIOURAL_ANSWER_WORD_THRESHOLD:
            is_substantial = word_count > BEHAVIOURAL_SUBSTANTIAL_WORD_THRESHOLD

    classification = {
        "response_type": response_type,
        "clarification_type": clarification_type,
        "is_substantial": is_substantial,
        "question_doubt_response": (
            result.clarification_response
            if clarification_type == "question_doubt"
            else None
        ),
    }
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

    latest_evaluation: dict[str, Any] | None = None
    skill_streaks = {
        key: dict(value)
        for key, value in (state.get("skill_evaluation_streaks") or {}).items()
    }
    if (
        response_type == "answer"
        and is_substantial
        and state.get("current_section_kind") == "technical"
        and result.answer_strength
    ):
        latest_evaluation = {"strength": result.answer_strength}
        if state.get("current_technical_skill"):
            key = str(state["current_technical_skill"]).casefold()
            current = dict(
                skill_streaks.get(key) or {"weak": 0, "adequate": 0, "strong": 0}
            )
            for strength in ("weak", "adequate", "strong"):
                current[strength] = (
                    int(current.get(strength) or 0) + 1
                    if strength == result.answer_strength
                    else 0
                )
            skill_streaks[key] = current

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
            {"live_evaluation": latest_evaluation, "evaluation_source": source}
        )
    pending_candidate_turn.update(
        {"response_type": response_type, "metadata": metadata}
    )

    base: dict[str, Any] = {
        "last_classification": classification,
        "classification_source": source,
        "last_response_type": response_type,
        "last_response_substantial": is_substantial,
        "last_skip_resume_skill_match": resume_skill_match,
        "latest_evaluation": latest_evaluation,
        "skill_evaluation_streaks": skill_streaks,
        "pending_candidate_turn": pending_candidate_turn,
        "violations_to_persist": new_violations,
        "recent_violations": recent_violations,
        "pregenerated_question": None,
        "pregenerated_closing_lead": None,
        "pending_clarification_text": None,
        "speculative_interviewer_result": None,
        **_self_intro_state_updates(state, text),
    }

    logger.info(
        "Interviewer turn resolved by merged model",
        extra={
            "candidate_assessment_id": state.get("candidate_assessment_id"),
            "response_type": response_type,
            "is_substantial": is_substantial,
            "answer_strength": result.answer_strength,
            "time_action": decision["action"],
            "classification_source": source,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
        },
    )

    forced_behavioural_preface = None
    if decision.get("transition_reason") == "behavioural_time_rescue":
        forced_behavioural_preface = (
            "Due to lack of time lets move to the behavioural section."
        )

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
            lead = str(result.acknowledgement or "").strip()
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

        question_text = str(result.question_text or "").strip()
        pregen: dict[str, Any] | None = None
        if question_text:
            acknowledgement = str(result.acknowledgement or "").strip()
            if forced_behavioural_preface:
                acknowledgement = forced_behavioural_preface
            elif is_deterministic_classification_source(source):
                acknowledgement = ""
            topic = str(result.topic or "").strip()
            if target["kind"] == "technical":
                strength_key = (
                    result.answer_strength
                    if (plan and result.answer_strength in plan)
                    else "adequate"
                )
                chosen = (plan or {}).get(strength_key) or {
                    "difficulty": None,
                    "probe_deeper": False,
                    "follow_interesting_thread": False,
                }
                follow_thread = bool(chosen.get("follow_interesting_thread"))
                pregen = {
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
                        topic=str(pregen["topic"]),
                        skill=skill_name,
                        asked_questions=questions_for_skill(state, skill_name),
                        used_topics=list(
                            (state.get("used_topics_by_skill") or {}).get(
                                skill_name.casefold(),
                                [],
                            )
                        ),
                        allow_related_probe=bool(pregen["probe_deeper"]),
                    )
                except ValueError as exc:
                    logger.info(
                        "Pregenerated technical question rejected: %s",
                        exc,
                    )
                    pregen = None
            else:
                pregen = {
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
                        topic=str(pregen["topic"]),
                        skill="",
                        asked_questions=behavioural_questions(state),
                        used_topics=[
                            str(item.get("topic") or "")
                            for item in behavioural_questions(state)
                            if item.get("topic")
                        ],
                        allow_related_probe=False,
                    )
                except ValueError as exc:
                    logger.info(
                        "Pregenerated behavioural question rejected: %s",
                        exc,
                    )
                    pregen = None

        if pregen:
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
                acknowledgement=str(pregen.get("acknowledgement") or "").strip(),
                question_text=str(pregen["question_text"]).strip(),
                topic=str(pregen.get("topic") or "").strip(),
                current_skill=(
                    str(pregen.get("skill") or "") or None
                    if target["kind"] == "technical"
                    else None
                ),
                current_difficulty=pregen.get("difficulty"),
                probe_deeper=bool(pregen.get("probe_deeper")),
                follow_interesting_thread=bool(pregen.get("follow_interesting_thread")),
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
            "pregenerated_question": pregen,
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


async def interviewer_turn(state: InterviewState) -> dict[str, Any]:
    """
    Classify, evaluate, and respond to one candidate utterance in a single pass.

    Unambiguous silence and short clarifications are answered instantly from
    templates. Everything else uses one merged structured call that routes the
    utterance and, when it is a substantial answer, produces the next question at a
    deterministically planned difficulty and section.

    Args:
        state: The current interview state.

    Returns:
        State updates with classification, optional staged question/closing lead,
        violations, and the routing key ``next_action``.
    """

    started_at = time.perf_counter()
    text = str(state.get("previous_candidate_response") or "")
    det = _deterministic_classification(state, text)

    if det is not None and _should_finish_deterministically(det):
        return _finish_deterministic(state, det, started_at)

    if is_self_intro_phase(state):
        # Self-intro routing is fully deterministic: avoid LLM misclassification
        # and use cumulative spoken-word count to determine substantiality.
        deterministic_result = InterviewerTurnResponse(
            response_type="answer",
            clarification_type=None,
            is_substantial=self_intro_is_substantial(state, text),
            answer_strength=None,
            acknowledgement=None,
            question_text=None,
            topic=None,
            clarification_response=None,
        )
        decision = decide_time_action(state)
        target = _resolve_target(state, decision)
        plan = (
            difficulty_plan(
                state,
                skill=str(
                    target["skill"] or state.get("current_technical_skill") or "skill"
                ),
                entering_new_section=bool(target["entering_new_section"]),
            )
            if target["kind"] == "technical"
            else None
        )
        return _apply_result(
            state,
            deterministic_result,
            decision,
            target,
            plan,
            "deterministic_self_intro",
            started_at,
        )

    if (
        state.get("current_section_kind") == "behavioural_cultural"
        and _spoken_word_count(text) > BEHAVIOURAL_ANSWER_WORD_THRESHOLD
    ):
        word_count = _spoken_word_count(text)
        deterministic_result = InterviewerTurnResponse(
            response_type="answer",
            clarification_type=None,
            is_substantial=(word_count > BEHAVIOURAL_SUBSTANTIAL_WORD_THRESHOLD),
            answer_strength=None,
            acknowledgement=None,
            question_text=None,
            topic=None,
            clarification_response=None,
        )
        decision = decide_time_action(state)
        target = _resolve_target(state, decision)
        return _apply_result(
            state,
            deterministic_result,
            decision,
            target,
            None,
            "deterministic_behavioural",
            started_at,
        )

    decision = decide_time_action(state)
    target = _resolve_target(state, decision)
    must_close = decision["action"] == "close"
    ask_next_question = not must_close
    plan = (
        difficulty_plan(
            state,
            skill=str(
                target["skill"] or state.get("current_technical_skill") or "skill"
            ),
            entering_new_section=bool(target["entering_new_section"]),
        )
        if target["kind"] == "technical"
        else None
    )

    cached_raw = state.get("speculative_interviewer_result")
    if cached_raw:
        try:
            cached_result = InterviewerTurnResponse(**cached_raw)
            return _apply_result(
                state,
                cached_result,
                decision,
                target,
                plan,
                "speculative_cache",
                started_at,
            )
        except Exception:
            logger.warning(
                "Speculative interviewer cache unusable; calling model",
                exc_info=True,
                extra={
                    "candidate_assessment_id": state.get("candidate_assessment_id"),
                },
            )

    messages = _interviewer_messages(
        state,
        target=target,
        decision=decision,
        ask_next_question=ask_next_question,
        must_close=must_close,
        plan=plan,
    )

    source = "llm"
    try:
        result = await interviewer_turn_with_schema(
            messages,
            InterviewerTurnResponse,
            key_slot=active_turn_key_slot(state),
        )
    except Exception:
        logger.exception(
            "Merged interviewer-turn call failed; using safe fallback",
            extra={
                "candidate_assessment_id": state.get("candidate_assessment_id"),
                "question_id": state.get("current_question_id"),
            },
        )
        # Treat the utterance as a substantial answer with neutral strength and let
        # the question generator regenerate, mirroring the legacy fallback path.
        result = _safe_merged_fallback(state, text, det)
        source = "safe_fallback"

    return _apply_result(
        state,
        result,
        decision,
        target,
        plan,
        source,
        started_at,
    )
