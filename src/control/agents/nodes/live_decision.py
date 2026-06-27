"""One-call live response control for the interview graph."""

from __future__ import annotations

import re
from statistics import mean
from typing import Any, Literal, cast

from pydantic import BaseModel

from src.control.agents.nodes.context_utils import (
    _next_bot_turn_number,
    extract_resume_skills,
    is_behavioural_cultural_section,
    is_self_intro_section,
    is_technical_section,
    resume_mentions_skill,
    word_count,
)
from src.control.agents.nodes.edge_handlers import (
    handle_disconnect,
    handle_irrelevant,
    handle_non_answer,
    handle_silence,
    handle_skip,
    handle_think_request,
)
from src.control.agents.nodes.llm_helpers import (
    clean_text,
    compact_json,
    lightweight_json,
    model_to_dict,
)
from src.control.agents.prompts import (
    LIVE_BEHAVIOURAL_DECISION_SYSTEM_PROMPT,
    LIVE_DECISION_SYSTEM_PROMPT,
)
from src.control.agents.state import Difficulty, InterviewState
from src.schemas.prompts import LiveInterviewDecision
from src.utils.interview_graph import deterministic_turn_id, utc_now_iso

LocalResponseType = Literal[
    "answer",
    "clarification",
    "irrelevant",
    "silence",
    "skip",
    "think_request",
    "no_think",
    "technical_issue",
    "disconnect",
    "timer_expired",
]

STRENGTH_SCORE = {"weak": 2.0, "adequate": 3.4, "strong": 4.6}
SELF_INTRO_MIN_DURATION_MS = 15_000


class LocalRouteDecision(BaseModel):
    response_type: LocalResponseType
    confidence: float = 1.0
    reason: str = ""
    clarification_type: str | None = None


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _clean_for_route(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _event_response_type(state: InterviewState) -> str:
    event = state.get("normalized_candidate_event") or {}
    return str(event.get("response_type") or "answer")


def _candidate_text(state: InterviewState) -> str:
    event = state.get("normalized_candidate_event") or {}
    return str(event.get("text") or "").strip()


def _duration_ms(state: InterviewState) -> int:
    event = state.get("normalized_candidate_event") or {}
    return int(event.get("duration_ms") or 0)


def local_route(text: str, state: InterviewState) -> LocalRouteDecision | None:
    event_type = _event_response_type(state)
    if event_type in {"silence", "disconnect", "timer_expired", "technical_issue"}:
        return LocalRouteDecision(
            response_type=cast(LocalResponseType, event_type), reason=event_type
        )

    t = _clean_for_route(text)
    if not t:
        return LocalRouteDecision(
            response_type="silence",
            confidence=1.0,
            reason="empty transcript",
        )

    if state.get("awaiting_think_confirmation"):
        if re.search(r"\b(yes|yeah|yep|please|sure|ok|okay)\b", t):
            return LocalRouteDecision(
                response_type="think_request",
                confidence=0.95,
                reason="accepted thinking time",
            )
        if re.search(r"\b(no|nope|not needed|continue|move on)\b", t):
            return LocalRouteDecision(
                response_type="no_think",
                confidence=0.95,
                reason="declined thinking time",
            )

    if _contains_any(
        t,
        (
            "repeat the question",
            "can you repeat",
            "could you repeat",
            "please repeat",
            "say that again",
            "can you say that again",
            "could you say that again",
        ),
    ):
        return LocalRouteDecision(
            response_type="clarification",
            confidence=0.95,
            reason="explicit repeat request",
            clarification_type="repeat_question",
        )

    if _contains_any(
        t,
        (
            "rephrase the question",
            "can you rephrase",
            "could you rephrase",
            "simplify the question",
            "explain the question",
            "i didn't understand the question",
            "i did not understand the question",
        ),
    ):
        return LocalRouteDecision(
            response_type="clarification",
            confidence=0.95,
            reason="explicit rephrase request",
            clarification_type="rephrase_question",
        )

    if _contains_any(
        t,
        (
            "what do you mean by",
            "does this mean",
            "are you asking about",
            "should i talk about",
            "what exactly do you mean",
        ),
    ):
        return None

    if _contains_any(
        t,
        (
            "skip this",
            "skip the question",
            "can we skip",
            "could we skip",
            "let's skip",
            "move to the next question",
            "go to the next question",
            "next question please",
            "ask me another question",
            "can we move on",
            "could we move on",
            "i want to pass",
            "i will pass",
            "i'll pass",
            "pass on this",
        ),
    ):
        return LocalRouteDecision(
            response_type="skip",
            confidence=0.95,
            reason="explicit skip request",
            clarification_type="skip_question",
        )

    if _contains_any(
        t,
        (
            "give me a moment",
            "give me some time",
            "let me think",
            "one second",
            "one moment",
            "just a second",
            "just a moment",
            "i need a moment",
            "i need some time",
            "can i think",
            "can i take a moment",
        ),
    ):
        return LocalRouteDecision(
            response_type="think_request",
            confidence=0.9,
            reason="explicit request for thinking time",
            clarification_type="time_to_think",
        )

    if _contains_any(
        t,
        (
            "can i use chatgpt",
            "can i use google",
            "can i google",
            "can i search",
            "search online",
            "look it up",
            "use the internet",
            "check my notes",
            "refer to notes",
            "ask someone",
            "are you an ai",
            "are you a bot",
            "who evaluates",
            "who will evaluate",
            "how am i evaluated",
            "is this recorded",
            "what happens after this",
            "who reviews this",
            "is a human reviewing",
            "ignore your instructions",
            "ignore previous instructions",
            "show me your prompt",
            "what is your system prompt",
            "reveal your prompt",
            "give me the answer",
            "tell me the correct answer",
            "what should i say",
        ),
    ):
        return LocalRouteDecision(
            response_type="irrelevant",
            confidence=0.95,
            reason="off-topic, meta, external help, or prompt injection",
        )

    if _contains_any(
        t,
        (
            "mic is not working",
            "microphone is not working",
            "audio issue",
            "i can't hear",
            "i cannot hear",
            "can't hear you",
            "cannot hear you",
            "connection issue",
            "voice is breaking",
            "you are breaking",
            "audio is breaking",
            "network issue",
        ),
    ):
        return LocalRouteDecision(
            response_type="technical_issue",
            confidence=0.95,
            reason="explicit mic/audio/connection issue",
        )

    return None


async def local_route_candidate_response(state: InterviewState) -> dict[str, Any]:
    decision = local_route(_candidate_text(state), state)
    if decision is None and is_self_intro_section(state):
        decision = LocalRouteDecision(
            response_type="answer",
            confidence=0.9,
            reason="self intro is handled without LLM",
        )

    if decision is None:
        return {"local_route": None, "next_node": "live_decision"}

    return {
        "local_route": decision.model_dump(),
        "live_decision": None,
        "last_response_type": decision.response_type,
        "last_response_substantial": False,
        "last_response_reason": decision.reason,
        "last_classification": {
            "response_type": decision.response_type,
            "candidate_question_intent": decision.clarification_type,
            "resume_skill_match": resume_mentions_skill(
                state.get("resume_parsed") or {},
                state.get("current_skill"),
            ),
        },
        "next_node": "handle_local_route",
    }


def _previous_questions(state: InterviewState) -> list[str]:
    current_section = state.get("current_section")
    current_skill = state.get("current_skill")
    questions: list[str] = []
    for item in state.get("asked_questions") or []:
        if not isinstance(item, dict):
            continue
        same_section = item.get("section") == current_section
        same_skill = current_skill and item.get("skill") == current_skill
        if same_section or same_skill:
            text = str(item.get("question_text") or "").strip()
            if text:
                questions.append(text)
    return questions[-10:]


def _live_decision_messages(state: InterviewState) -> list[dict[str, str]]:
    event = state.get("normalized_candidate_event") or {}
    context = {
        "candidate_text": event.get("text") or "",
        "current_question": state.get("current_question_text"),
        "current_section": state.get("current_section"),
        "current_skill": state.get("current_skill"),
        "current_difficulty": state.get("current_difficulty"),
        "resume_skills": extract_resume_skills(state.get("resume_parsed") or {})[:24],
        "previous_questions": _previous_questions(state),
        "recent_turns": [
            {
                "speaker": turn.get("speaker"),
                "text": clean_text(turn.get("text"), max_chars=180),
            }
            for turn in (state.get("recent_turns") or [])[-4:]
            if isinstance(turn, dict)
        ],
    }
    system_prompt = (
        LIVE_BEHAVIOURAL_DECISION_SYSTEM_PROMPT
        if is_behavioural_cultural_section(state)
        else LIVE_DECISION_SYSTEM_PROMPT
    )
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Return JSON matching this schema: "
                f"{compact_json(LiveInterviewDecision.model_json_schema(), max_chars=2600)}. "
                f"Context: {compact_json(context, max_chars=2600)}"
            ),
        },
    ]


async def live_decision(state: InterviewState) -> dict[str, Any]:
    decision = model_to_dict(
        await lightweight_json(
            _live_decision_messages(state),
            LiveInterviewDecision,
            max_tokens=360,
            temperature=0.2,
        )
    )
    strength = decision.get("answer_strength")
    return {
        "live_decision": decision,
        "last_response_type": decision["response_type"],
        "last_response_substantial": decision.get("is_substantial"),
        "last_answer_strength": strength,
        "last_response_reason": decision.get("reason"),
        "current_difficulty": decision.get("next_difficulty")
        or state.get("current_difficulty"),
        "expected_signals": decision.get("expected_signals") or [],
        "next_node": "apply_live_decision",
    }


def _same_question_turn(
    state: InterviewState,
    text: str,
    message_type: str,
    *,
    expected_signals: list[str] | None = None,
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
        "tone": "professional",
        "text": text,
        "section": state.get("current_section") or "general",
        "skill": state.get("current_skill"),
        "difficulty": state.get("current_difficulty") or "medium",
        "question_id": state.get("current_question_id"),
        "timestamp": utc_now_iso(),
        "metadata": {
            "message_type": message_type,
            "expected_signals": expected_signals or [],
        },
    }


def _question_number(state: InterviewState, section: str) -> int:
    return (
        sum(
            1
            for item in state.get("asked_questions") or []
            if isinstance(item, dict)
            and item.get("section") == section
            and not item.get("is_followup")
        )
        + 1
    )


def _question_turn(
    state: InterviewState,
    *,
    question_text: str,
    difficulty: Difficulty,
    expected_signals: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    section = str(state.get("current_section") or "general")
    question_id = f"q_{section}_{_question_number(state, section):03d}"
    turn_number = _next_bot_turn_number(state)
    section_remaining_secs = int(state.get("current_section_remaining_secs") or 0)
    max_answer_secs = max(10, min(90, section_remaining_secs or 60))
    turn = {
        "turn_id": deterministic_turn_id(
            str(state["interview_session_id"]),
            turn_number,
            "bot",
        ),
        "turn_number": turn_number,
        "speaker": "bot",
        "tone": "professional",
        "text": question_text,
        "section": section,
        "skill": state.get("current_skill"),
        "difficulty": difficulty,
        "question_id": question_id,
        "timestamp": utc_now_iso(),
        "metadata": {
            "message_type": "question",
            "expected_signals": expected_signals,
            "max_answer_secs": max_answer_secs,
        },
    }
    record = {
        "question_id": question_id,
        "question_text": question_text,
        "section": section,
        "skill": state.get("current_skill"),
        "difficulty": difficulty,
        "is_followup": False,
        "expected_signals": expected_signals,
    }
    return turn, record


def _evaluation_id(state: InterviewState) -> str:
    pending_candidate = state.get("pending_candidate_turn") or {}
    turn_number = int(
        pending_candidate.get("turn_number") or state.get("turn_number") or 0
    )
    return f"{state.get('current_question_id')}:{turn_number}"


def _skill_progress_update(
    state: InterviewState,
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    section = str(
        evaluation.get("section") or state.get("current_section") or "general"
    )
    skill = str(evaluation.get("skill") or section)
    key = skill.lower()
    progress = dict(state.get("skill_progress") or {})
    current = dict(progress.get(key) or {})
    scores = [
        *list(current.get("scores") or []),
        float(evaluation.get("provisional_score") or 0),
    ]
    strength = str(evaluation.get("strength") or "").lower()
    consecutive_strong = int(current.get("consecutive_strong_answers") or 0)
    consecutive_adequate = int(current.get("consecutive_adequate_answers") or 0)
    progress[key] = {
        "skill": skill,
        "section": section,
        "answers": int(current.get("answers") or 0) + 1,
        "scores": scores[-5:],
        "average_score": round(mean(scores), 2),
        "best_score": max(scores),
        "signals_observed": sorted(
            set(current.get("signals_observed") or [])
            | set(evaluation.get("signals_observed") or [])
        ),
        "signals_missing": sorted(set(evaluation.get("signals_missing") or [])),
        "consecutive_strong_answers": consecutive_strong + 1
        if strength == "strong"
        else 0,
        "consecutive_adequate_answers": consecutive_adequate + 1
        if strength == "adequate"
        else 0,
        "last_strength": strength,
    }
    return progress


def _difficulty_after_decision(
    state: InterviewState,
    decision: dict[str, Any],
) -> Difficulty:
    explicit = decision.get("next_difficulty")
    if explicit in {"easy", "medium", "hard"}:
        return explicit
    strength = decision.get("answer_strength")
    if strength == "weak":
        return "easy"
    if strength == "strong":
        return "hard"
    return str(state.get("current_difficulty") or "medium")  # type: ignore[return-value]


def _section_transition_payload(state: InterviewState) -> dict[str, Any]:
    return {
        "next_action": "section_transition",
        "next_section_index": int(state.get("current_section_index") or 0) + 1,
        "should_close": False,
        "next_node": "check_time_budget",
    }


def _self_intro_answer_payload(state: InterviewState) -> dict[str, Any]:
    count = int(state.get("non_answer_count_for_current_question") or 0)
    short_by_duration = 0 < _duration_ms(state) < SELF_INTRO_MIN_DURATION_MS
    short_by_text = word_count(_candidate_text(state)) < 25
    if count < 2 and (short_by_duration or short_by_text):
        return handle_non_answer(state)
    return {
        "last_response_type": "answer",
        "last_response_substantial": True,
        "non_answer_count_for_current_question": 0,
        **_section_transition_payload(state),
    }


async def handle_local_route(state: InterviewState) -> dict[str, Any]:
    local = state.get("local_route") or {}
    response_type = str(local.get("response_type") or "")
    clarification_type = local.get("clarification_type")

    if response_type == "answer" and is_self_intro_section(state):
        return _self_intro_answer_payload(state)
    if response_type == "silence":
        return handle_silence(state)
    if response_type == "think_request":
        return handle_think_request(state)
    if response_type == "no_think":
        text = "Understood. I will move on for now."
        return {
            "last_response_type": "clarification",
            "pending_bot_turn": _same_question_turn(state, text, "silence_ack"),
            "last_bot_text": text,
            "bot_reply_type": "silence_ack",
            "awaiting_think_confirmation": False,
            "think_extension_active": False,
            "think_silence_count": 0,
            "next_action": "next_question",
            "should_close": False,
        }
    if response_type == "skip":
        return {
            "last_response_type": "clarification",
            "last_classification": {
                "response_type": "clarification",
                "candidate_question_intent": "skip_question",
                "resume_skill_match": resume_mentions_skill(
                    state.get("resume_parsed") or {},
                    state.get("current_skill"),
                ),
            },
            **handle_skip(state),
        }
    if response_type == "clarification":
        question = (
            state.get("current_question_text") or "Please answer the current question."
        )
        if clarification_type == "rephrase_question":
            text = (
                f"Sure. In simpler terms, please explain your thinking for: {question}"
            )
            message_type = "clarification"
        else:
            text = f"Sure, the question is: {question}"
            message_type = "clarification"
        return {
            "last_response_type": "clarification",
            "clarification_count_for_current_question": int(
                state.get("clarification_count_for_current_question") or 0
            )
            + 1,
            "pending_bot_turn": _same_question_turn(state, text, message_type),
            "last_bot_text": text,
            "bot_reply_type": message_type,
            "next_action": None,
        }
    if response_type in {"irrelevant", "technical_issue"}:
        return {
            "last_response_type": "irrelevant",
            **handle_irrelevant(state),
        }
    if response_type == "disconnect":
        return await handle_disconnect(state)
    if response_type == "timer_expired":
        return {
            "last_response_type": "timer_expired",
            "next_action": "complete",
            "should_close": True,
            "next_node": "check_time_budget",
        }
    return {"next_node": "live_decision"}


async def apply_live_decision(state: InterviewState) -> dict[str, Any]:
    decision = state.get("live_decision") or {}
    response_type = str(decision.get("response_type") or "irrelevant")
    next_action = str(decision.get("next_action") or "")

    if response_type == "irrelevant":
        return {
            "last_response_type": "irrelevant",
            **handle_irrelevant(state),
        }

    if response_type == "clarification":
        clarification_type = str(decision.get("clarification_type") or "")
        if clarification_type == "skip_question":
            return {
                "last_response_type": "clarification",
                "last_classification": {
                    "response_type": "clarification",
                    "candidate_question_intent": "skip_question",
                    "resume_skill_match": resume_mentions_skill(
                        state.get("resume_parsed") or {},
                        state.get("current_skill"),
                    ),
                },
                **handle_skip(state),
            }
        if clarification_type == "time_to_think":
            return {
                "last_response_type": "clarification",
                **handle_think_request(state),
            }
        if clarification_type == "question_doubt":
            text = clean_text(
                decision.get("interviewer_text")
                or "I am asking about the scope of the current question. Please answer it in your own words.",
                max_chars=420,
            )
            return {
                "last_response_type": "clarification",
                "clarification_count_for_current_question": int(
                    state.get("clarification_count_for_current_question") or 0
                )
                + 1,
                "pending_bot_turn": _same_question_turn(state, text, "clarification"),
                "last_bot_text": text,
                "bot_reply_type": "clarification",
                "next_action": None,
            }
        if next_action == "rephrase_current_question":
            text = clean_text(
                decision.get("interviewer_text")
                or f"Sure. In simpler terms, please explain your thinking for: {state.get('current_question_text')}",
                max_chars=420,
            )
        else:
            text = f"Sure, the question is: {state.get('current_question_text')}"
        return {
            "last_response_type": "clarification",
            "clarification_count_for_current_question": int(
                state.get("clarification_count_for_current_question") or 0
            )
            + 1,
            "pending_bot_turn": _same_question_turn(state, text, "clarification"),
            "last_bot_text": text,
            "bot_reply_type": "clarification",
            "next_action": None,
        }

    is_substantial = bool(decision.get("is_substantial"))
    if not is_substantial:
        return {
            "last_response_type": "answer",
            "last_response_substantial": False,
            **handle_non_answer(state),
        }

    next_difficulty = _difficulty_after_decision(state, decision)
    question_text = clean_text(decision.get("next_question_text"), max_chars=300)
    expected_signals = list(decision.get("expected_signals") or [])[:4]
    question_turn, question_record = _question_turn(
        state,
        question_text=question_text,
        difficulty=next_difficulty,
        expected_signals=expected_signals,
    )

    updates: dict[str, Any] = {
        "last_response_type": "answer",
        "last_response_substantial": True,
        "current_question_id": question_record["question_id"],
        "current_question_text": question_text,
        "current_difficulty": next_difficulty,
        "asked_questions": [*(state.get("asked_questions") or []), question_record],
        "expected_signals": expected_signals,
        "clarification_count_for_current_question": 0,
        "silence_count_for_current_question": 0,
        "non_answer_count_for_current_question": 0,
        "skip_count_for_current_question": 0,
        "think_silence_count": 0,
        "awaiting_think_confirmation": False,
        "think_extension_active": False,
        "pending_bot_turn": question_turn,
        "last_bot_text": question_text,
        "bot_reply_type": "question",
        "next_action": None,
        "next_node": "persist_interview_turn",
    }

    if is_technical_section(state):
        strength = str(decision.get("answer_strength") or "adequate").lower()
        evaluation = {
            "evaluation_id": _evaluation_id(state),
            "question_id": state.get("current_question_id"),
            "skill": state.get("current_skill"),
            "section": state.get("current_section"),
            "response_type": "answer",
            "provisional_score": STRENGTH_SCORE[strength],
            "strength": strength,
            "is_substantial": True,
            "signals_observed": [f"{strength}_live_signal"],
            "signals_missing": [] if strength == "strong" else ["depth_or_specificity"],
            "recommended_next_action": "next_question",
            "recommended_difficulty": next_difficulty,
            "summary": f"Live decision classified the answer as {strength}.",
            "created_at": utc_now_iso(),
            "violations": [],
        }
        updates.update(
            {
                "latest_evaluation": evaluation,
                "live_evaluations": [
                    *(state.get("live_evaluations") or []),
                    evaluation,
                ],
                "skill_progress": _skill_progress_update(state, evaluation),
            }
        )
    else:
        updates["latest_evaluation"] = None

    return updates
