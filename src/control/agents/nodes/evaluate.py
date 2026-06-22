"""Live answer evaluation node."""

from __future__ import annotations

import logging
import time

from src.control.agents.llm import groq_complete, parse_json
from src.control.agents.prompts import EVALUATION_SYSTEM, build_evaluation_prompt
from src.control.agents.state import InterviewState, QuestionScore, Violation
from src.control.time_manager import next_difficulty_from_quality

logger = logging.getLogger(__name__)


def _fallback_quality(answer: str) -> dict:
    words = len(answer.split())
    if words < 3:
        return {
            "quality": "non_answer",
            "score": 1.0,
            "signals_present": [],
            "signals_missing": ["No meaningful answer provided"],
            "is_substantial": False,
            "key_concept_demonstrated": "",
            "one_line_feedback": "I could not evaluate a substantive answer.",
            "reasoning": "The answer was too short to evaluate.",
        }
    if words < 18:
        return {
            "quality": "weak",
            "score": 4.0,
            "signals_present": ["Attempted to answer"],
            "signals_missing": ["Needs more detail and specificity"],
            "is_substantial": False,
            "key_concept_demonstrated": "partial_answer",
            "one_line_feedback": "The answer was relevant but too brief.",
            "reasoning": "The answer addressed the question only at a high level.",
        }
    return {
        "quality": "adequate",
        "score": 6.0,
        "signals_present": ["Provided a relevant answer"],
        "signals_missing": ["Could include more concrete evidence"],
        "is_substantial": True,
        "key_concept_demonstrated": "general_understanding",
        "one_line_feedback": "The answer was relevant and usable for follow-up.",
        "reasoning": "Fallback evaluation based on answer length and relevance.",
    }


def _normalize_eval(raw: dict, answer: str) -> dict:
    fallback = _fallback_quality(answer)
    result = {**fallback, **(raw or {})}
    quality = str(result.get("quality") or fallback["quality"]).lower()
    if quality not in {"strong", "adequate", "weak", "non_answer"}:
        score = float(result.get("score") or fallback["score"])
        quality = (
            "strong"
            if score >= 8
            else "adequate"
            if score >= 5
            else "weak"
            if score >= 3
            else "non_answer"
        )
    score = max(0.0, min(10.0, float(result.get("score") or fallback["score"])))
    result["quality"] = quality
    result["score"] = score
    result["raw_score"] = score
    result["signals_present"] = list(
        result.get("signals_present") or result.get("signals_demonstrated") or []
    )
    result["signals_missing"] = list(result.get("signals_missing") or [])
    result["signals_demonstrated"] = result["signals_present"]
    result["is_substantial"] = bool(
        result.get("is_substantial", quality in {"adequate", "strong"})
    )
    result["key_concept_demonstrated"] = str(
        result.get("key_concept_demonstrated") or ""
    ).strip()
    result["one_line_feedback"] = str(
        result.get("one_line_feedback") or fallback["one_line_feedback"]
    )
    result["reasoning"] = str(result.get("reasoning") or fallback["reasoning"])
    return result


async def evaluate_answer(state: InterviewState) -> dict:
    answer = (
        state.get("candidate_raw_text") or state.get("last_transcript") or ""
    ).strip()
    sections = state.get("sections") or []
    idx = int(state.get("current_section_index") or 0)
    section = sections[idx] if idx < len(sections) else {}
    section_name = str(section.get("section_name") or section.get("name") or "general")
    skill = section.get("skill")

    try:
        response = await groq_complete(
            system=EVALUATION_SYSTEM,
            user=build_evaluation_prompt(state, answer),
            json_mode=True,
            max_tokens=650,
            temperature=0.2,
        )
        eval_result = _normalize_eval(parse_json(response), answer)
    except Exception:
        logger.exception("Answer evaluation failed; using fallback evaluation")
        eval_result = _normalize_eval({}, answer)

    concept = (
        eval_result.get("key_concept_demonstrated")
        or state.get("current_question_concept")
        or skill
        or section_name
    )
    eval_result["key_concept_demonstrated"] = str(concept)
    eval_result["question"] = state.get("current_question_text") or state.get(
        "current_question", ""
    )
    eval_result["section"] = section_name
    eval_result["skill"] = skill
    eval_result["difficulty"] = state.get("current_question_difficulty", "medium")
    eval_result["turn_number"] = int(state.get("turn_number") or 0)
    eval_result["candidate_answer"] = answer
    eval_result["evaluated_at"] = time.time()

    q_score: QuestionScore = {
        "question": eval_result["question"],
        "section": section_name,
        "skill": skill,
        "concept": str(concept),
        "difficulty": eval_result["difficulty"],
        "quality": eval_result["quality"],
        "raw_score": float(eval_result["score"]),
        "reasoning": eval_result["reasoning"],
        "signals_demonstrated": list(eval_result.get("signals_present") or []),
        "signals_missing": list(eval_result.get("signals_missing") or []),
        "turn_number": int(state.get("turn_number") or 0),
    }

    violations: list[Violation] = []
    if eval_result.get("resume_contradiction"):
        violations.append(
            {
                "turn_number": int(state.get("turn_number") or 0),
                "violation_type": "resume_mismatch",
                "candidate_transcript": answer,
                "timestamp": time.time(),
            }
        )
    if eval_result.get("yoe_contradiction"):
        violations.append(
            {
                "turn_number": int(state.get("turn_number") or 0),
                "violation_type": "yoe_mismatch",
                "candidate_transcript": answer,
                "timestamp": time.time(),
            }
        )

    quality = str(eval_result.get("quality") or "adequate")
    was_retry = bool(state.get("last_question_was_weak_retry"))
    next_question_mode = "normal"
    if quality in {"weak", "non_answer"} and not was_retry:
        next_question_mode = "easier_same_concept"

    priority = section.get("priority_score")
    next_level = next_difficulty_from_quality(
        int(state.get("current_difficulty_level") or 2), quality, priority
    )
    if next_question_mode == "easier_same_concept":
        next_level = max(1, next_level - 1)

    return {
        "last_evaluation": eval_result,
        "question_scores": [q_score],
        "answer_evaluations": [eval_result],
        "violations": violations,
        "current_difficulty_level": next_level,
        "next_question_mode": next_question_mode,
        "last_question_was_weak_retry": False,
        "candidate_raw_text": "",
        "candidate_stt_confidence": None,
        "response_class": None,
        "silence_state": {},
        "next_node": "persist_turn",
    }
