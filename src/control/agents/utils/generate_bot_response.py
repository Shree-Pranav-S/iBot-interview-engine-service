"""Helpers for static bot speech, transitions, and question rephrases."""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any

from src.control.agents.key_routing import active_turn_key_slot
from src.control.agents.prompts import QUESTION_REPHRASE_SYSTEM_PROMPT
from src.control.agents.state import InterviewState
from src.control.agents.utils.question_diversity import normalize_question_text
from src.control.agents.utils.question_strategy import current_section_name
from src.control.agents.utils.turn_utils import build_bot_turn
from src.core.exceptions.interview import QuestionValidationException
from src.core.services import llm_service
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
        "current_section": current_section_name(state),
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
        result = await llm_service.lightweight(
            messages,
            QuestionRephraseResponse,
            key_slot=active_turn_key_slot(state),
            max_tokens=128,
            temperature=0.1,
        )
        rewritten = " ".join(result.question_text.split())
        similarity = SequenceMatcher(
            None,
            normalize_question_text(question),
            normalize_question_text(rewritten),
        ).ratio()
        if similarity >= 0.9:
            raise QuestionValidationException(
                "rephrased question is too close to original"
            )
        previous_rephrase = str(state.get("last_rephrased_question") or "").strip()
        if previous_rephrase:
            previous_similarity = SequenceMatcher(
                None,
                normalize_question_text(previous_rephrase),
                normalize_question_text(rewritten),
            ).ratio()
            if previous_similarity >= 0.88:
                raise QuestionValidationException(
                    "rephrased question repeats the previous rewrite"
                )
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
