"""Generate static bot speech and dynamic-question transition prefaces."""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any

from src.control.agents.key_routing import active_turn_key_slot
from src.control.agents.nodes.classify_response import (
    is_deterministic_classification_source,
)
from src.control.agents.nodes.llm_helpers import rephrase_with_schema
from src.control.agents.nodes.turn_utils import build_bot_turn
from src.control.agents.prompts import QUESTION_REPHRASE_SYSTEM_PROMPT
from src.control.agents.state import InterviewState
from src.control.agents.templates import choose_template
from src.schemas.prompts import QuestionRephraseResponse

logger = logging.getLogger(__name__)


def _reply(
    state: InterviewState,
    text: str,
    reply_type: str,
    *,
    question_type: str,
    next_action: str = "await_candidate_response",
    **updates: Any,
) -> dict[str, Any]:
    """
    Construct a complete bot turn that immediately yields control back to LiveKit.

    Args:
        state: The current interview state.
        text: The text the bot should speak.
        reply_type: The internal categorization of the response.
        question_type: The tracking metadata type for the turn.
        next_action: The next node to route to (usually waiting for the candidate).
        **updates: Additional state updates to apply.

    Returns:
        A dictionary of state updates containing the `pending_bot_turn`.
    """
    clean_text = " ".join(text.split())
    pending_bot_turn = build_bot_turn(
        state,
        text=clean_text,
        question_type=question_type,
    )
    return {
        "bot_reply_text": clean_text,
        "bot_reply_type": reply_type,
        "pending_bot_turn": pending_bot_turn,
        "next_action": next_action,
        "phase_complete": False,
        "pending_clarification_text": None,
        **updates,
    }


def _continue_with_question(
    text: str,
    reply_type: str,
) -> dict[str, Any]:
    """Let the dynamic generator attach the next question to this preface."""

    return {
        "bot_reply_text": "",
        "bot_reply_type": reply_type,
        "pending_bot_turn": None,
        "response_preface_text": " ".join(text.split()),
        "next_action": "generate_question",
        "should_advance_question": True,
        "phase_complete": False,
        "silence_stage": "none",
        "pending_clarification_text": None,
    }


def _normalized_question(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9+#.]+", value.casefold()))


def _fallback_rephrase(
    question: str,
    *,
    use_alternate: bool = False,
) -> str:
    """
    Provide a meaning-preserving rewrite if the fast model is unavailable.
    Uses regex replacements to manually alter common question structures.

    Args:
        question: The original question text.
        use_alternate: Whether to use the secondary rewrite pattern (if the first was already used).

    Returns:
        The rewritten question.
    """

    original = " ".join(question.split()).rstrip(" ?")
    transformations = (
        (
            r"(?i)^how would you\s+(.+)$",
            (
                r"Which steps would you follow to \1?"
                if use_alternate
                else r"What approach would you take to \1?"
            ),
        ),
        (
            r"(?i)^what is the (?:primary )?purpose of\s+(.+)$",
            (
                r"What role does \1 play in practice?"
                if use_alternate
                else r"In practical terms, what does \1 help accomplish?"
            ),
        ),
        (
            r"(?i)^what is the difference between\s+(.+)\s+and\s+(.+)$",
            (
                r"In practical use, how do \1 and \2 differ?"
                if use_alternate
                else r"How would you distinguish \1 from \2?"
            ),
        ),
        (
            r"(?i)^tell me about\s+(.+)$",
            (
                r"What specific experience can you share involving \1?"
                if use_alternate
                else r"Could you walk me through \1?"
            ),
        ),
        (
            r"(?i)^(?:can|could) you (?:describe|explain|share)\s+(.+)$",
            (
                r"What example would best illustrate \1?"
                if use_alternate
                else r"How would you explain \1 in your own words?"
            ),
        ),
        (
            r"(?i)^what is\s+(.+)$",
            (
                r"What does \1 mean in a practical setting?"
                if use_alternate
                else r"How would you explain \1 in practical terms?"
            ),
        ),
    )
    for pattern, replacement in transformations:
        rewritten = re.sub(pattern, replacement, original).strip()
        if rewritten != original:
            return rewritten
    return f"How would you explain your approach to this topic: {original}?"


async def _rephrase_question(
    state: InterviewState,
    question: str,
) -> str:
    """
    Use an LLM to rephrase a question without changing its core technical intent.
    Validates that the rewritten question is structurally different enough from the original.

    Args:
        state: The current interview state.
        question: The question to rewrite.

    Returns:
        The rewritten string.
    """
    context = {
        "original_question": question,
        "previous_rephrase": state.get("last_rephrased_question"),
        "current_section": state.get("current_section"),
        "current_skill": state.get("current_technical_skill"),
        "question_difficulty": state.get("current_question_difficulty"),
    }
    messages = [
        {"role": "system", "content": QUESTION_REPHRASE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, default=str),
        },
    ]
    try:
        result = await rephrase_with_schema(
            messages,
            QuestionRephraseResponse,
            key_slot=active_turn_key_slot(state),
        )
        rewritten = " ".join(result.question_text.split())
        similarity = SequenceMatcher(
            None,
            _normalized_question(question),
            _normalized_question(rewritten),
        ).ratio()
        if similarity >= 0.9:
            raise ValueError("rephrased question is too close to original")
        previous_rephrase = str(state.get("last_rephrased_question") or "").strip()
        if previous_rephrase:
            previous_similarity = SequenceMatcher(
                None,
                _normalized_question(previous_rephrase),
                _normalized_question(rewritten),
            ).ratio()
            if previous_similarity >= 0.88:
                raise ValueError("rephrased question repeats the previous rewrite")
        return rewritten
    except Exception as exc:
        logger.warning(
            "Question rephrase failed; using local fallback: %s",
            exc,
            extra={
                "candidate_assessment_id": state.get("candidate_assessment_id"),
                "question_id": state.get("current_question_id"),
            },
        )
    return _fallback_rephrase(
        question,
        use_alternate=bool(state.get("last_rephrased_question")),
    )


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
    is_intro = bool(state.get("is_self_introduction"))
    question = (
        state.get("current_question_text")
        or "Could you answer the current interview question?"
    )
    intro_repeat_text = (
        state.get("last_spoken_opening_text")
        or state.get("last_rephrased_question")
        or question
    )

    if response_type == "silence":
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

    if response_type == "answer":
        if not bool(state.get("last_response_substantial")):
            is_intro = bool(state.get("is_self_introduction"))
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
        repeatable_question = (
            intro_repeat_text
            if is_intro
            else (state.get("last_rephrased_question") or question)
        )
        return _reply(
            state,
            choose_template(
                "repeat_question",
                question=repeatable_question,
            ),
            "question_repeat",
            question_type="clarification_response",
            silence_stage="none",
        )
    if clarification_type == "rephrase_question":
        # The merged interviewer-turn call may have already produced the rephrase in
        # the same round trip; only call the rephrase model when it did not.
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
