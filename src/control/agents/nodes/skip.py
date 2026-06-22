"""Skip handling node."""

from __future__ import annotations

import time

from src.control.agents.prompts import trim_to_2_sentences
from src.control.agents.state import InterviewState, QuestionScore


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


async def handle_skip(state: InterviewState) -> dict:
    turn_number = int(state.get("turn_number") or 0) + 1
    if not state.get("skip_requested"):
        text = trim_to_2_sentences(
            "I understand you'd like to move on, but please give at least a partial answer. Even a high-level overview is helpful."
        )
        return {
            "last_bot_text": text,
            "bot_reply_text": text,
            "bot_reply_type": "nudge",
            "turn_number": turn_number,
            "transcript": [_bot_turn(state, turn_number, text, "nudge")],
            "candidate_raw_text": "",
            "response_class": None,
            "next_node": "await_response",
        }

    text = "Understood, we'll move on. Let's try a different question."
    q_score: QuestionScore = {
        "question": state.get("current_question_text")
        or state.get("current_question", ""),
        "section": state.get("current_section_name", "general"),
        "skill": None,
        "concept": state.get("current_question_concept"),
        "difficulty": state.get("current_question_difficulty", "medium"),
        "quality": "non_answer",
        "raw_score": 2.0,
        "reasoning": "Candidate explicitly skipped the question.",
        "signals_demonstrated": [],
        "signals_missing": ["Question skipped"],
        "turn_number": int(state.get("turn_number") or 0),
    }
    eval_result = {
        "quality": "non_answer",
        "score": 2.0,
        "raw_score": 2.0,
        "reasoning": q_score["reasoning"],
        "question": q_score["question"],
        "section": q_score["section"],
        "turn_number": q_score["turn_number"],
        "signals_present": [],
        "signals_missing": ["Question skipped"],
        "is_substantial": False,
        "key_concept_demonstrated": "",
    }
    return {
        "last_bot_text": text,
        "bot_reply_text": text,
        "bot_reply_type": "skip",
        "turn_number": turn_number,
        "transcript": [_bot_turn(state, turn_number, text, "skip")],
        "question_scores": [q_score],
        "answer_evaluations": [eval_result],
        "last_evaluation": eval_result,
        "candidate_raw_text": "",
        "response_class": None,
        "current_difficulty_level": max(
            1, int(state.get("current_difficulty_level") or 2) - 1
        ),
        "next_node": "generate_question",
    }
