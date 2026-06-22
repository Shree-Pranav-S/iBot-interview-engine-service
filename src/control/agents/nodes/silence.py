"""Silence and think-time handling node."""

from __future__ import annotations

import time

from src.control.agents.prompts import trim_to_2_sentences
from src.control.agents.state import InterviewState, QuestionScore, Violation


def _bot_turn(state: InterviewState, text: str, turn_type: str) -> tuple[int, dict]:
    turn_number = int(state.get("turn_number") or 0) + 1
    return turn_number, {
        "turn_number": turn_number,
        "speaker": "bot",
        "text": text,
        "tone": "neutral",
        "section": state.get("current_section_name", "general"),
        "turn_type": turn_type,
        "type": turn_type,
        "question": state.get("current_question_text")
        or state.get("current_question", ""),
        "timestamp": time.time(),
    }


def _zero_score(
    state: InterviewState, reason: str
) -> tuple[QuestionScore, Violation, dict]:
    q_score: QuestionScore = {
        "question": state.get("current_question_text")
        or state.get("current_question", ""),
        "section": state.get("current_section_name", "general"),
        "skill": None,
        "concept": state.get("current_question_concept"),
        "difficulty": state.get("current_question_difficulty", "medium"),
        "quality": "non_answer",
        "raw_score": 0.0,
        "reasoning": reason,
        "signals_demonstrated": [],
        "signals_missing": ["No response provided"],
        "turn_number": int(state.get("turn_number") or 0),
    }
    violation: Violation = {
        "turn_number": int(state.get("turn_number") or 0),
        "violation_type": "silence",
        "candidate_transcript": "",
        "timestamp": time.time(),
    }
    eval_result = {
        "quality": "non_answer",
        "score": 0.0,
        "raw_score": 0.0,
        "reasoning": reason,
        "question": q_score["question"],
        "section": q_score["section"],
        "turn_number": q_score["turn_number"],
        "signals_present": [],
        "signals_missing": ["No response provided"],
        "is_substantial": False,
        "key_concept_demonstrated": "",
    }
    return q_score, violation, eval_result


def _reset_silence() -> dict:
    return {
        "silence_state": {},
    }


async def handle_silence(state: InterviewState) -> dict:
    response_class = state.get("response_class")
    awaiting_decision = bool(state.get("awaiting_think_decision"))
    think_active = bool(state.get("think_timer_active"))
    candidate_text = (state.get("candidate_raw_text") or "").strip()

    if response_class == "think_request":
        text = trim_to_2_sentences(
            "Sure, take 15 seconds to think. Please share your answer when you are ready."
        )
        turn_number, turn = _bot_turn(state, text, "think_timer")
        return {
            "last_bot_text": text,
            "bot_reply_text": text,
            "bot_reply_type": "think_timer",
            "turn_number": turn_number,
            "transcript": [turn],
            "candidate_raw_text": "",
            "response_class": None,
            "silence_state": {},
            "next_node": "await_response",
        }

    if awaiting_decision:
        reason = "No answer after think-time prompt."
        text = trim_to_2_sentences(
            "No problem. Since I did not get an answer, I'll move on to the next question."
        )
        q_score, violation, eval_result = _zero_score(state, reason)
        turn_number, turn = _bot_turn(state, text, "silence_ack")
        patch = {
            "last_bot_text": text,
            "bot_reply_text": text,
            "bot_reply_type": "silence_ack",
            "turn_number": turn_number,
            "transcript": [turn],
            "question_scores": [q_score],
            "answer_evaluations": [eval_result],
            "last_evaluation": eval_result,
            "violations": [violation],
            "candidate_raw_text": "",
            "response_class": None,
            "next_question_mode": "normal",
            "next_node": "generate_question",
        }
        patch.update(_reset_silence())
        return patch

    if think_active:
        reason = "No answer after 15 second thinking window."
        text = trim_to_2_sentences(
            "I still could not detect an answer, so I'll move on to the next question."
        )
        q_score, violation, eval_result = _zero_score(state, reason)
        turn_number, turn = _bot_turn(state, text, "silence_ack")
        patch = {
            "last_bot_text": text,
            "bot_reply_text": text,
            "bot_reply_type": "silence_ack",
            "turn_number": turn_number,
            "transcript": [turn],
            "question_scores": [q_score],
            "answer_evaluations": [eval_result],
            "last_evaluation": eval_result,
            "violations": [violation],
            "candidate_raw_text": "",
            "response_class": None,
            "next_question_mode": "normal",
            "next_node": "generate_question",
        }
        patch.update(_reset_silence())
        return patch

    if candidate_text and candidate_text != "__SILENCE__":
        reason = (
            "Candidate declined or did not provide an answer after think-time prompt."
        )
        text = trim_to_2_sentences(
            "Understood. Since there is no answer to evaluate, I'll move on to the next question."
        )
        q_score, violation, eval_result = _zero_score(state, reason)
        turn_number, turn = _bot_turn(state, text, "silence_ack")
        patch = {
            "last_bot_text": text,
            "bot_reply_text": text,
            "bot_reply_type": "silence_ack",
            "turn_number": turn_number,
            "transcript": [turn],
            "question_scores": [q_score],
            "answer_evaluations": [eval_result],
            "last_evaluation": eval_result,
            "violations": [violation],
            "candidate_raw_text": "",
            "response_class": None,
            "next_node": "generate_question",
        }
        patch.update(_reset_silence())
        return patch

    text = "Do you need some time to think?"
    turn_number, turn = _bot_turn(state, text, "think_offer")
    return {
        "last_bot_text": text,
        "bot_reply_text": text,
        "bot_reply_type": "think_offer",
        "turn_number": turn_number,
        "transcript": [turn],
        "candidate_raw_text": "",
        "response_class": None,
        "silence_state": {},
        "next_node": "await_response",
    }
