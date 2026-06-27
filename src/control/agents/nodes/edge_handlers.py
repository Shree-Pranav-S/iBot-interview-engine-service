"""Static edge handlers for the simplified interview graph."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from src.control.agents.nodes.context_utils import (
    _next_bot_turn_number,
    is_self_intro_section,
    resume_mentions_skill,
)
from src.control.agents.state import InterviewState
from src.data.repositories import interview_session_repository
from src.utils.interview_graph import (
    deterministic_turn_id,
    deterministic_violation_id,
    utc_now_iso,
)

GRACE_PERIOD_SECS = 300

ELABORATION_RESPONSES = (
    "Could you elaborate a little more so I have enough detail to assess your answer?",
    "Please add a bit more detail, including your reasoning or a concrete example.",
    "That is too brief to evaluate. Could you expand on it?",
    "Could you walk me through your thinking in a little more detail?",
    "Please continue with more specifics about your approach.",
    "I need a fuller answer here. Could you explain it further?",
    "Could you add context, the key steps, and the outcome or trade-off?",
    "Please give a more complete answer rather than a short summary.",
    "Could you share an example or more reasoning behind that?",
    "Please expand your answer enough for me to understand your experience.",
)
SELF_INTRO_ELABORATION_RESPONSES = (
    "Could you share a little more about your background, experience, and role fit?",
    "Please expand your introduction with your recent work and strongest skills.",
    "That was quite brief. Could you tell me more about your professional background?",
    "Could you add more context about your experience and relevant projects?",
    "Please take a bit more time with your introduction and include your strengths.",
    "Could you walk me through your background in a little more detail?",
    "Please include the kind of work you have done recently and what you are strongest at.",
    "Could you say more about your journey and the areas where you have hands-on experience?",
    "I would like a more complete introduction before we move on. Please continue.",
    "Please add a few more details about your experience and why this role fits you.",
)
REPEAT_RESPONSES = (
    "Sure, the question is: {question}",
    "Of course. The question was: {question}",
    "No problem. Let me repeat it: {question}",
    "Certainly. Here is the question again: {question}",
    "Yes. Please answer this question: {question}",
    "Sure. I asked: {question}",
    "Let me say it again: {question}",
    "Absolutely. The current question is: {question}",
    "Here it is again: {question}",
    "I will repeat the question: {question}",
)
REPHRASE_RESPONSES = (
    "Sure. In simpler terms, please answer this in your own words: {question}",
    "Of course. Think of it as a practical explanation of this question: {question}",
    "No problem. I am asking for your approach and reasoning for: {question}",
    "Certainly. Please explain the main idea behind this question: {question}",
    "Let me simplify it. Share what you would do and why for: {question}",
    "Sure. Focus on the practical steps you would take for: {question}",
    "I can rephrase it this way: explain your thinking for: {question}",
    "Yes. Please describe the approach you would take for: {question}",
    "In other words, walk me through your understanding of: {question}",
    "Put simply, tell me how you would handle this: {question}",
)
SKIP_PARTIAL_RESPONSES = (
    "Could you try giving a partial attempt before we move on?",
    "Please share whatever part of the answer you can attempt.",
    "Even partial reasoning is useful here. Could you try briefly?",
    "Before skipping, please give a short attempt if you can.",
    "Could you answer the part you are most comfortable with?",
    "Please try a brief answer, even if it is incomplete.",
    "Let us give it one attempt first. What can you say about it?",
    "Could you share your initial thinking before we move on?",
    "Please try explaining any relevant idea or example you know.",
    "I can move on after this, but first please make a partial attempt.",
)
SKIP_RESUME_MATCH_PREFIXES = (
    "It is surprising you want to skip this considering you have mentioned "
    "{skill} in your background.",
    "Since {skill} appears in your background, I would like you to try at "
    "least part of this.",
    "You have listed {skill} in your background, so please give this one a "
    "brief attempt.",
    "Given that {skill} is mentioned in your profile, it would be helpful to "
    "hear even a partial answer.",
    "Because your background includes {skill}, I want to give you a chance to "
    "attempt this before we move on.",
    "I noticed {skill} in your background, so please try sharing what you can.",
    "Since you have shown {skill} as part of your experience, a short attempt "
    "would still be useful.",
    "Your background mentions {skill}, so I would prefer not to skip this "
    "without a quick attempt.",
    "Considering {skill} is part of your profile, please try answering the "
    "part you know.",
    "As {skill} is included in your background, even a partial explanation "
    "would help here.",
)
IRRELEVANT_RESPONSES = (
    "Let's stay with the interview question. Please answer the question I asked.",
    "I need to keep us on topic. Please respond to the current question.",
    "Let's return to the interview. Please answer the current question.",
    "That does not address the question. Please focus on your answer.",
    "Please keep your response relevant to the current interview question.",
    "Let's continue with the assessment question in front of us.",
    "I cannot use that as an answer. Please respond to the question asked.",
    "Please bring your answer back to the topic of the current question.",
    "We need to stay on track. Please answer the interview question.",
    "That is outside the scope of this interview question. Please answer it directly.",
)
THINK_WAIT_RESPONSES = (
    "Sure, take your time.",
    "Of course. Take a moment.",
    "No problem, take a moment to think.",
    "Sure. I will give you a little time.",
)
THINK_NUDGE_RESPONSES = (
    "Please share your answer now so we can continue.",
    "Whenever you are ready, please give your answer now.",
    "Let's continue. Please share what you can.",
)


def _candidate_turn_number(state: InterviewState) -> int:
    pending_candidate = state.get("pending_candidate_turn")
    if pending_candidate:
        return int(pending_candidate.get("turn_number") or 0)
    return int(state.get("turn_number") or 0)


def _turn(
    state: InterviewState,
    text: str,
    message_type: str,
    *,
    tone: str = "professional",
) -> dict[str, Any]:
    turn_number = _next_bot_turn_number(state)
    return {
        "turn_id": deterministic_turn_id(
            str(state["interview_session_id"]),
            turn_number,
            "bot",
        ),
        "turn_number": turn_number,
        "speaker": "bot",
        "tone": tone,
        "text": text,
        "section": state.get("current_section") or "general",
        "skill": state.get("current_skill"),
        "difficulty": state.get("current_difficulty") or "medium",
        "question_id": state.get("current_question_id"),
        "timestamp": utc_now_iso(),
        "metadata": {"message_type": message_type},
    }


def _pick(options: tuple[str, ...], state: InterviewState, salt: str) -> str:
    basis = f"{state.get('current_question_id')}-{state.get('turn_number')}-{salt}"
    index = sum(ord(char) for char in basis) % len(options)
    return options[index]


def _same_question_payload(
    state: InterviewState,
    text: str,
    message_type: str,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "pending_bot_turn": _turn(state, text, message_type),
        "last_bot_text": text,
        "bot_reply_type": message_type,
        "next_action": None,
        "awaiting_think_confirmation": False,
        "think_extension_active": False,
        "think_silence_count": 0,
    }
    payload.update(extra)
    return payload


def _minimal_progress(
    state: InterviewState,
    *,
    response_type: str,
    missing_signal: str,
    score: float = 1.0,
) -> dict[str, Any]:
    skill = str(state.get("current_skill") or state.get("current_section") or "general")
    key = skill.lower()
    progress = dict(state.get("skill_progress") or {})
    current = dict(progress.get(key) or {})
    scores = [*list(current.get("scores") or []), score]
    progress[key] = {
        "skill": skill,
        "section": state.get("current_section") or "general",
        "answers": int(current.get("answers") or 0) + 1,
        "scores": scores[-5:],
        "average_score": round(sum(scores) / len(scores), 2),
        "best_score": max(scores),
        "signals_observed": list(current.get("signals_observed") or []),
        "signals_missing": [missing_signal],
        "consecutive_strong_answers": 0,
        "consecutive_adequate_answers": 0,
        "last_strength": "weak",
    }
    evaluation = {
        "evaluation_id": (
            f"{state.get('current_question_id')}:{_candidate_turn_number(state)}:"
            f"{response_type}"
        ),
        "question_id": state.get("current_question_id"),
        "skill": state.get("current_skill"),
        "section": state.get("current_section"),
        "response_type": response_type,
        "provisional_score": score,
        "strength": "weak",
        "is_substantial": False,
        "signals_observed": [],
        "signals_missing": [missing_signal],
        "recommended_next_action": "next_question",
        "recommended_difficulty": "easy",
        "summary": missing_signal,
        "created_at": utc_now_iso(),
    }
    return {
        "skill_progress": progress,
        "latest_evaluation": evaluation,
        "live_evaluations": [*(state.get("live_evaluations") or []), evaluation],
        "current_difficulty": "easy",
    }


def _violation(
    state: InterviewState,
    violation_type: str,
    *,
    severity: str,
    candidate_transcript: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    turn_number = _candidate_turn_number(state)
    return {
        "violation_id": deterministic_violation_id(
            str(state["interview_session_id"]),
            turn_number,
            violation_type,
        ),
        "turn_number": turn_number,
        "violation_type": violation_type,
        "candidate_transcript": candidate_transcript,
        "severity": severity,
        "timestamp": utc_now_iso(),
        "metadata": metadata or {},
    }


async def handle_clarification(state: InterviewState) -> dict[str, Any]:
    count = int(state.get("clarification_count_for_current_question") or 0) + 1
    question = (
        state.get("current_question_text") or "Please answer the current question."
    )
    classification = state.get("last_classification") or {}
    intent = str(classification.get("candidate_question_intent") or "").strip()

    if intent == "skip_question":
        return handle_skip(state)

    if intent == "rephrase_question":
        text = _pick(REPHRASE_RESPONSES, state, f"rephrase-{count}").format(
            question=question
        )
    else:
        text = _pick(REPEAT_RESPONSES, state, f"repeat-{count}").format(
            question=question
        )
    return _same_question_payload(
        state,
        text,
        "clarification",
        clarification_count_for_current_question=count,
    )


def handle_silence(state: InterviewState) -> dict[str, Any]:
    count = int(state.get("silence_count_for_current_question") or 0) + 1

    if state.get("think_extension_active"):
        think_count = int(state.get("think_silence_count") or 0) + 1
        if think_count == 1:
            prompt = _pick(THINK_NUDGE_RESPONSES, state, "think-nudge")
            return _same_question_payload(
                state,
                prompt,
                "nudge",
                silence_count_for_current_question=count,
                think_silence_count=think_count,
                think_extension_active=True,
            )

        text = "Understood. I will move on for now."
        return {
            **_minimal_progress(
                state,
                response_type="silence",
                missing_signal="no_response_after_thinking_time",
            ),
            "pending_bot_turn": _turn(state, text, "silence_ack"),
            "last_bot_text": text,
            "bot_reply_type": "silence_ack",
            "silence_count_for_current_question": count,
            "think_silence_count": think_count,
            "think_extension_active": False,
            "awaiting_think_confirmation": False,
            "violation_to_persist": _violation(
                state,
                "no_response_after_thinking_time",
                severity="medium",
                metadata={"question_id": state.get("current_question_id")},
            ),
            "next_action": "next_question",
            "should_close": False,
        }

    if count == 1:
        return _same_question_payload(
            state,
            "Do you need some time to think?",
            "think_offer",
            silence_count_for_current_question=count,
            awaiting_think_confirmation=True,
            think_extension_active=False,
            think_silence_count=0,
        )

    text = "Understood. I will move on for now."
    return {
        **_minimal_progress(
            state,
            response_type="silence",
            missing_signal="no_response_after_silence_prompt",
        ),
        "pending_bot_turn": _turn(state, text, "silence_ack"),
        "last_bot_text": text,
        "bot_reply_type": "silence_ack",
        "silence_count_for_current_question": count,
        "awaiting_think_confirmation": False,
        "think_extension_active": False,
        "violation_to_persist": _violation(
            state,
            "no_response_after_silence_prompt",
            severity="medium",
            metadata={"question_id": state.get("current_question_id")},
        ),
        "next_action": "next_question",
        "should_close": False,
    }


def handle_think_request(state: InterviewState) -> dict[str, Any]:
    prompt = _pick(THINK_WAIT_RESPONSES, state, "think-wait")
    return _same_question_payload(
        state,
        prompt,
        "think_wait",
        awaiting_think_confirmation=False,
        think_extension_active=True,
        think_silence_count=0,
    )


def handle_non_answer(state: InterviewState) -> dict[str, Any]:
    count = int(state.get("non_answer_count_for_current_question") or 0) + 1

    if int(state.get("skip_count_for_current_question") or 0) > 0:
        return {
            **_minimal_progress(
                state,
                response_type="answer",
                missing_signal="no_partial_attempt_after_skip_prompt",
            ),
            "non_answer_count_for_current_question": count,
            "next_action": "next_question",
            "should_close": False,
        }

    if count <= 2:
        options = (
            SELF_INTRO_ELABORATION_RESPONSES
            if is_self_intro_section(state)
            else ELABORATION_RESPONSES
        )
        prompt = _pick(options, state, f"elaborate-{count}")
        return _same_question_payload(
            state,
            prompt,
            "redirect",
            non_answer_count_for_current_question=count,
        )

    return {
        **_minimal_progress(
            state,
            response_type="answer",
            missing_signal="insufficient_response_after_elaboration_prompt",
        ),
        "non_answer_count_for_current_question": count,
        "next_action": "next_question",
        "should_close": False,
    }


def handle_irrelevant(state: InterviewState) -> dict[str, Any]:
    count = int(state.get("irrelevant_count_total") or 0) + 1
    prompt = _pick(IRRELEVANT_RESPONSES, state, f"irrelevant-{count}")
    return _same_question_payload(
        state,
        prompt,
        "redirect",
        irrelevant_count_total=count,
    )


def handle_skip(state: InterviewState) -> dict[str, Any]:
    count = int(state.get("skip_count_for_current_question") or 0) + 1
    classification = state.get("last_classification") or {}
    skill = state.get("current_skill")
    resume_has_skill = bool(
        classification.get("resume_skill_match")
    ) or resume_mentions_skill(
        state.get("resume_parsed") or {},
        skill,
    )

    if count == 1:
        prompt = _pick(SKIP_PARTIAL_RESPONSES, state, "skip-partial")
        if resume_has_skill and skill:
            prefix = _pick(SKIP_RESUME_MATCH_PREFIXES, state, "skip-resume").format(
                skill=skill
            )
            prompt = f"{prefix} {prompt}"
        return _same_question_payload(
            state,
            prompt,
            "redirect",
            skip_count_for_current_question=count,
        )

    return {
        **_minimal_progress(
            state,
            response_type="skip",
            missing_signal="candidate_skipped_after_partial_attempt_prompt",
        ),
        "skip_count_for_current_question": count,
        "next_action": "next_question",
        "should_close": False,
    }


async def handle_disconnect(state: InterviewState) -> dict[str, Any]:
    session_id = state.get("interview_session_id")
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=GRACE_PERIOD_SECS)
    if session_id:
        await interview_session_repository.pause_session(str(session_id), expires_at)

    return {
        "session_status": "PAUSED",
        "disconnected_at": now.isoformat(),
        "grace_period_expires_at": expires_at.isoformat(),
        "next_node": "end",
        "next_action": "paused",
        "should_close": False,
    }
