"""Irrelevant response handling node."""

from __future__ import annotations

import time

from src.control.agents.prompts import trim_to_2_sentences
from src.control.agents.state import InterviewState, QuestionScore, Violation


def _score(state: InterviewState, reason: str) -> QuestionScore:
    return {
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
        "signals_missing": ["Did not address the question"],
        "turn_number": int(state.get("turn_number") or 0),
    }


def _bot_turn(
    state: InterviewState, turn_number: int, text: str, turn_type: str
) -> dict:
    return {
        "turn_number": turn_number,
        "speaker": "bot",
        "text": text,
        "tone": "neutral",
        "section": state.get("current_section_name", "general"),
        "turn_type": turn_type,
        "type": turn_type,
        "timestamp": time.time(),
    }


async def handle_irrelevant(state: InterviewState) -> dict:
    new_count = (
        int(state.get("irrelevant_strike_count") or state.get("irrelevant_count") or 0)
        + 1
    )
    candidate_text = (
        state.get("candidate_raw_text") or state.get("last_transcript") or ""
    )
    question = state.get("current_question_text") or state.get("current_question", "")
    reason = f"Irrelevant response strike {new_count}."
    violation: Violation = {
        "turn_number": int(state.get("turn_number") or 0),
        "violation_type": "irrelevant",
        "candidate_transcript": candidate_text,
        "timestamp": time.time(),
    }
    q_score = _score(state, reason)
    eval_result = {
        "quality": "non_answer",
        "score": 0.0,
        "raw_score": 0.0,
        "reasoning": reason,
        "question": question,
        "section": q_score["section"],
        "turn_number": q_score["turn_number"],
        "signals_present": [],
        "signals_missing": ["Did not address the question"],
        "is_substantial": False,
        "key_concept_demonstrated": "",
    }

    turn_number = int(state.get("turn_number") or 0) + 1
    if new_count >= 3:
        text = trim_to_2_sentences(
            "This interview has been ended because of repeated off-topic responses. Thank you for your time."
        )
        termination: Violation = {
            "turn_number": int(state.get("turn_number") or 0),
            "violation_type": "terminated",
            "candidate_transcript": candidate_text,
            "timestamp": time.time(),
        }
        return {
            "last_bot_text": text,
            "bot_reply_text": text,
            "bot_reply_type": "closing",
            "turn_number": turn_number,
            "transcript": [_bot_turn(state, turn_number, text, "closing")],
            "violations": [violation, termination],
            "question_scores": [q_score],
            "answer_evaluations": [eval_result],
            "last_evaluation": eval_result,
            "irrelevant_count": new_count,
            "irrelevant_strike_count": new_count,
            "session_status": "TERMINATED",
            "should_close": True,
            "candidate_raw_text": "",
            "response_class": None,
            "next_node": "closing",
        }

    text = (
        f"Let's stay focused on the interview. Please answer this question directly: {question}"
        if new_count == 1
        else f"I need you to stay on topic. This is the final warning. Please answer: {question}"
    )
    text = trim_to_2_sentences(text)
    return {
        "last_bot_text": text,
        "bot_reply_text": text,
        "bot_reply_type": "irrelevant_warning",
        "turn_number": turn_number,
        "transcript": [_bot_turn(state, turn_number, text, "irrelevant_warning")],
        "violations": [violation],
        "irrelevant_count": new_count,
        "irrelevant_strike_count": new_count,
        "question_scores": [q_score],
        "answer_evaluations": [eval_result],
        "last_evaluation": eval_result,
        "candidate_raw_text": "",
        "response_class": None,
        "next_node": "await_response",
    }
