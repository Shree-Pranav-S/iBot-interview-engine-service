"""Generate static bot speech and dynamic-question transition prefaces."""

from __future__ import annotations

from typing import Any

from src.control.agents.state import InterviewState
from src.control.agents.templates import choose_template
from src.control.agents.utils.classify_response import (
    is_deterministic_classification_source,
    self_intro_is_substantial,
)
from src.control.agents.utils.generate_bot_response import (
    _continue_with_question,
    _rephrase_question,
    _reply,
)
from src.control.agents.utils.question_strategy import is_self_intro_phase


async def generate_bot_response(state: InterviewState) -> dict[str, Any]:
    """
    Select static speech or use a bounded LLM-produced doubt clarification.

    This node handles all bot speech that is NOT a new dynamic question (e.g.,
    handling silence, irrelevant answers, requests to repeat, skips). It either
    replies and waits, or generates a preface and routes to `generate_question`.

    Args:
        state: The current interview state.

    Returns:
        State updates containing the bot reply and next routing action.
    """

    response_type = state.get("last_response_type")
    classification = state.get("last_classification") or {}
    is_intro = is_self_intro_phase(state)
    question = (
        state.get("current_question_text")
        or "Could you answer the current interview question?"
    )

    if response_type == "silence":
        if is_intro and self_intro_is_substantial(
            state,
            str(state.get("previous_candidate_response") or ""),
        ):
            return _continue_with_question("", "question_transition")
        stage = state.get("silence_stage") or "none"
        if stage == "none":
            return _reply(
                state,
                choose_template("silence_offer"),
                "think_offer",
                question_type="silence_response",
                silence_stage="awaiting_think_confirmation",
            )
        if stage == "awaiting_think_confirmation":
            if is_intro:
                return _reply(
                    state,
                    choose_template("self_intro_elaborate"),
                    "elaboration_request",
                    question_type="elaboration_request",
                    self_intro_elaboration_requested=True,
                    silence_stage="none",
                )
            return _continue_with_question(
                choose_template("no_response_move_on"),
                "silence_ack",
            )
        if stage == "thinking":
            return _reply(
                state,
                choose_template("think_nudge"),
                "think_nudge",
                question_type="silence_response",
                silence_stage="nudged",
            )
        if is_intro:
            return _reply(
                state,
                choose_template("self_intro_elaborate"),
                "elaboration_request",
                question_type="elaboration_request",
                self_intro_elaboration_requested=True,
                silence_stage="none",
            )
        return _continue_with_question(
            choose_template("no_response_move_on"),
            "silence_ack",
        )

    if response_type == "irrelevant":
        if is_intro:
            return _reply(
                state,
                choose_template("self_intro_elaborate"),
                "intro_stay_on_topic",
                question_type="elaboration_request",
                self_intro_elaboration_requested=True,
                silence_stage="none",
            )
        return _reply(
            state,
            choose_template("irrelevant_redirect"),
            "irrelevant_redirect",
            question_type="irrelevant_response",
            silence_stage="none",
        )

    if response_type == "interview_meta":
        meta_type = str(classification.get("interview_meta_type") or "")
        template_name = (
            "interview_meta_time_remaining"
            if meta_type == "time_remaining"
            else "interview_meta_guidance"
        )
        return _reply(
            state,
            choose_template(template_name),
            f"interview_meta_{meta_type or 'guidance'}",
            question_type="interview_meta_response",
            silence_stage="none",
        )

    if response_type == "answer":
        intro_text = str(state.get("previous_candidate_response") or "")
        if is_intro and self_intro_is_substantial(state, intro_text):
            return _continue_with_question("", "question_transition")
        if not bool(state.get("last_response_substantial")):
            return _reply(
                state,
                choose_template(
                    "self_intro_elaborate" if is_intro else "elaborate_answer"
                ),
                "elaboration_request",
                question_type="elaboration_request",
                self_intro_elaboration_requested=(
                    bool(state.get("self_intro_elaboration_requested")) or is_intro
                ),
                silence_stage="none",
            )
        if is_deterministic_classification_source(
            str(state.get("classification_source") or "")
        ):
            return _continue_with_question("", "question_transition")
        return _continue_with_question(
            choose_template("substantial_acknowledgement"),
            "question_transition",
        )

    clarification_type = str(classification.get("clarification_type") or "")
    if clarification_type == "repeat_question":
        return _reply(
            state,
            choose_template(
                "repeat_question",
                question=state.get("last_rephrased_question") or question,
            ),
            "question_repeat",
            question_type="clarification_response",
            silence_stage="none",
        )
    if clarification_type == "rephrase_question":
        # Stage two normally produces the rephrase; call the dedicated rephrase
        # fallback only when that response was unavailable.
        rephrased_question = str(
            state.get("pending_clarification_text") or ""
        ).strip() or await _rephrase_question(state, question)
        return _reply(
            state,
            choose_template(
                "rephrase_question",
                rephrased_question=rephrased_question,
            ),
            "question_rephrase",
            question_type="clarification_response",
            silence_stage="none",
            last_rephrased_question=rephrased_question,
        )
    if clarification_type == "time_to_think":
        return _reply(
            state,
            choose_template("think_wait"),
            "think_wait",
            question_type="clarification_response",
            silence_stage="thinking",
        )
    if clarification_type == "decline_think_time":
        return _reply(
            state,
            choose_template("think_declined"),
            "think_declined",
            question_type="clarification_response",
            silence_stage="none",
        )
    if clarification_type == "skip_question":
        attempts = int(state.get("skip_attempts_for_current_question") or 0) + 1
        if attempts == 1:
            partial_request = choose_template("skip_partial_attempt")
            if state.get("last_skip_resume_skill_match"):
                prefix = choose_template(
                    "skip_resume_skill_prefix",
                    skill=(state.get("current_technical_skill") or "this skill"),
                )
                partial_request = f"{prefix} {partial_request}"
            return _reply(
                state,
                partial_request,
                "skip_partial_request",
                question_type="skip_response",
                skip_attempts_for_current_question=attempts,
                silence_stage="none",
            )
        if is_intro:
            return _reply(
                state,
                choose_template("self_intro_elaborate"),
                "skip_intro_stay",
                question_type="elaboration_request",
                skip_attempts_for_current_question=attempts,
                self_intro_elaboration_requested=True,
                silence_stage="none",
            )
        return {
            **_continue_with_question(
                choose_template("skip_acknowledgement"),
                "skip_ack",
            ),
            "skip_attempts_for_current_question": attempts,
        }
    if clarification_type == "question_doubt":
        generated = (
            str(state.get("pending_clarification_text") or "").strip()
            or str(classification.get("question_doubt_response") or "").strip()
        )
        text = generated or choose_template(
            "question_doubt_fallback",
            question=question,
        )
        return _reply(
            state,
            text,
            "question_doubt_clarification",
            question_type="clarification_response",
            silence_stage="none",
        )

    return _reply(
        state,
        choose_template("irrelevant_redirect"),
        "irrelevant_redirect",
        question_type="irrelevant_response",
        silence_stage="none",
    )
