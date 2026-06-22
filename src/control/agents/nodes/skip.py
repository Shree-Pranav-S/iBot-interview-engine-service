"""Skip handling node."""

from __future__ import annotations

import time

from src.control.agents.prompts import trim_to_2_sentences
from src.control.agents.state import InterviewState, QuestionScore, Violation
from src.control.session_loader import extract_resume_skills


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

    sections = state.get("sections") or []
    idx = int(state.get("current_section_index") or 0)
    section = sections[idx] if idx < len(sections) else {}
    skill = section.get("skill")

    resume_skills = extract_resume_skills(
        state.get("resume_parsed") or state.get("resume_context") or {}
    )
    has_contradiction = False
    if skill and any(
        skill.lower() in rs.lower() or rs.lower() in skill.lower()
        for rs in resume_skills
    ):
        has_contradiction = True

    violations: list[Violation] = []
    if has_contradiction:
        violations.append(
            {
                "turn_number": turn_number,
                "violation_type": "resume_mismatch",
                "candidate_transcript": state.get("candidate_raw_text") or "",
                "timestamp": time.time(),
            }
        )

    if not state.get("skip_requested"):
        if has_contradiction:
            text = trim_to_2_sentences(
                f"It is surprising that you want to skip this, considering your background in {skill}. "
                "However, if you'd still like to move on, please let me know."
            )
        else:
            text = trim_to_2_sentences(
                "I understand you'd like to move on, but please give at least a partial answer. Even a high-level overview is helpful."
            )

        return {
            "last_bot_text": text,
            "bot_reply_text": text,
            "bot_reply_type": "nudge",
            "turn_number": turn_number,
            "transcript": [_bot_turn(state, turn_number, text, "nudge")],
            "violations": violations,
            "candidate_raw_text": "",
            "candidate_stt_confidence": None,
            "response_class": None,
            "awaiting_think_decision": False,
            "think_timer_active": False,
            "skip_requested": True,
            "next_node": "await_response",
        }

    if has_contradiction:
        text = "Understood. It is surprising that you have no experience to share here considering your background, but we will move on."
    else:
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
        "last_evaluation": eval_result,
        "violations": violations,
        "candidate_raw_text": "",
        "candidate_stt_confidence": None,
        "response_class": None,
        "awaiting_think_decision": False,
        "think_timer_active": False,
        "current_difficulty_level": max(
            1, int(state.get("current_difficulty_level") or 2) - 1
        ),
        "skip_requested": False,
        "next_node": "generate_question",
    }
