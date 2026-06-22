"""Question generation node."""

from __future__ import annotations

import logging
import time

from src.control.agents.llm import groq_complete, parse_json
from src.control.agents.state import InterviewState
from src.control.time_manager import (
    check_interview_complete,
    check_section_should_advance,
    compute_section_time_remaining,
    difficulty_label,
)

logger = logging.getLogger(__name__)

QUESTION_SYSTEM = """\
You are a senior interviewer conducting a live voice interview. Your words will be spoken aloud.

Rules:
- Sound warm, professional, and natural.
- Briefly acknowledge the candidate's previous answer before asking the next question.
- Do not reveal scores or say whether the answer was good or bad.
- Ask exactly one question.
- Keep it concise, 1-2 sentences maximum.
- Avoid markdown, bullets, and numbered lists.
- Do not repeat concepts already covered unless explicitly told this is an easier retry.

Return ONLY valid JSON:
{"question_text":"spoken interviewer utterance","concept_tag":"core_concept","difficulty":"easy|medium|hard"}
"""


def _section(state: InterviewState) -> dict:
    sections = state.get("sections") or []
    idx = int(state.get("current_section_index") or 0)
    return sections[idx] if idx < len(sections) else {}  # type: ignore


def _history(state: InterviewState) -> str:
    turns = state.get("transcript") or []
    recent = turns[-6:]
    if not recent:
        return "No conversation yet."
    lines: list[str] = []
    for turn in recent:
        speaker = "INTERVIEWER" if turn.get("speaker") == "bot" else "CANDIDATE"
        text = str(turn.get("text") or "")
        if len(text) > 350:
            text = text[:350] + "..."
        if text:
            lines.append(f"{speaker}: {text}")
    return "\n".join(lines) or "No conversation yet."


def _resume_summary(state: InterviewState) -> str:
    resume = state.get("resume_context") or state.get("resume_parsed") or {}
    parts: list[str] = []
    if resume.get("summary"):
        parts.append(str(resume["summary"]))
    if resume.get("experience_years"):
        parts.append(f"Experience: {resume['experience_years']} years")
    skills = resume.get("skills") or []
    if isinstance(skills, list) and skills:
        parts.append("Skills: " + ", ".join(str(item) for item in skills[:12]))
    return " | ".join(parts) if parts else "Not available"


def _fallback_question(state: InterviewState, section: dict, *, retry: bool) -> dict:
    name = str(section.get("section_name") or section.get("name") or "general")
    skill = section.get("skill") or name
    remaining = compute_section_time_remaining(state)
    concept = str(state.get("current_question_concept") or skill or name)
    difficulty = difficulty_label(int(state.get("current_difficulty_level") or 2))

    if retry:
        return {
            "question_text": f"Let's simplify that a bit. Can you give a basic example of how you would use {concept} in a real project?",
            "concept_tag": concept,
            "difficulty": "easy",
        }
    if remaining < 60:
        return {
            "question_text": f"Before we move on, what is one key point you would want me to remember about your experience with {skill}?",
            "concept_tag": f"{skill}_wrap_up",
            "difficulty": "easy",
        }
    if name == "self_intro":
        return {
            "question_text": "Could you briefly introduce yourself and share the experience most relevant to this role?",
            "concept_tag": "self_introduction",
            "difficulty": "easy",
        }
    if name.lower() in {"behavioural", "behavioral"}:
        return {
            "question_text": "Thanks for sharing that. Tell me about a time you faced a difficult problem at work and how you handled it.",
            "concept_tag": "problem_solving",
            "difficulty": "medium",
        }
    if name.lower() == "cultural":
        return {
            "question_text": "That's helpful context. What kind of team environment helps you do your best work?",
            "concept_tag": "team_environment",
            "difficulty": "easy",
        }
    return {
        "question_text": f"Thanks, I see the direction you were taking. Can you walk me through a practical project where you used {skill}, including one tradeoff you had to make?",
        "concept_tag": str(skill),
        "difficulty": difficulty,
    }


def _prompt(state: InterviewState, section: dict, retry: bool, difficulty: str) -> str:
    last_eval = state.get("last_evaluation") or {}
    used = state.get("used_concepts") or state.get("concepts_covered_in_section") or []
    remaining = int(compute_section_time_remaining(state))
    section_name = str(section.get("section_name") or section.get("name") or "general")
    skill = section.get("skill") or section_name
    retry_instruction = ""
    if retry:
        retry_instruction = (
            "The previous answer was weak or incomplete. Ask one easier question on the SAME concept, "
            "then the graph will move on to a different concept. Do not make it feel punitive."
        )
    elif remaining < 60:
        retry_instruction = "Less than one minute remains in this section. Ask a quick wrap-up question."
    elif remaining < 120:
        retry_instruction = (
            "Under two minutes remain in this section. Keep the next question focused."
        )
    else:
        retry_instruction = "Ask about a new concept within this section."

    return f"""
INTERVIEW CONTEXT
Role: {state.get("role_name") or "the role"}
Company: {state.get("company_name") or "the company"}
Current section: {section_name}
Skill: {skill}
Priority: {section.get("priority_score") or 5}/10
Time remaining in section: {remaining} seconds
Candidate background: {_resume_summary(state)}

RECENT CONVERSATION
{_history(state)}

LAST ANSWER EVALUATION
Quality: {last_eval.get("quality", "N/A")}
Score: {last_eval.get("score", last_eval.get("raw_score", "N/A"))}/10
Signals present: {", ".join(last_eval.get("signals_present") or last_eval.get("signals_demonstrated") or []) or "none"}
Signals missing: {", ".join(last_eval.get("signals_missing") or []) or "none"}
Internal feedback: {last_eval.get("one_line_feedback") or last_eval.get("reasoning") or "N/A"}

PROGRESS
Questions asked in this section: {state.get("questions_asked_in_section") or section.get("questions_asked") or 0}
Concepts already covered: {", ".join(str(item) for item in used) if used else "none"}
Target difficulty: {difficulty}

SPECIAL INSTRUCTIONS
{retry_instruction}
""".strip()


def _update_section_for_question(
    state: InterviewState, concept: str
) -> tuple[list[dict], list[str], int]:
    sections = [dict(section) for section in state.get("sections") or []]
    idx = int(state.get("current_section_index") or 0)
    used = list(
        state.get("used_concepts") or state.get("concepts_covered_in_section") or []
    )
    questions_asked = int(state.get("questions_asked_in_section") or 0) + 1
    if concept and concept not in used:
        used.append(concept)
    if idx < len(sections):
        section_concepts = list(sections[idx].get("concepts_covered") or [])  # type: ignore
        if concept and concept not in section_concepts:
            section_concepts.append(concept)
        sections[idx]["questions_asked"] = (
            int(sections[idx].get("questions_asked") or 0) + 1  # type: ignore
        )
        sections[idx]["concepts_covered"] = section_concepts
    return sections, used, questions_asked


async def generate_question(state: InterviewState) -> dict:
    if check_interview_complete(state):
        return {
            "should_close": True,
            "session_status": "COMPLETED",
            "next_node": "closing",
        }

    if check_section_should_advance(state):
        return {"next_node": "section_transition"}

    sections = state.get("sections") or []
    idx = int(state.get("current_section_index") or 0)
    if idx >= len(sections):
        return {
            "should_close": True,
            "session_status": "COMPLETED",
            "next_node": "closing",
        }

    section = sections[idx]
    retry = state.get("next_question_mode") == "easier_same_concept"
    fallback = _fallback_question(state, section, retry=retry)  # type: ignore
    difficulty = (
        "easy"
        if retry
        else difficulty_label(int(state.get("current_difficulty_level") or 2))
    )

    try:
        raw = await groq_complete(
            system=QUESTION_SYSTEM,
            user=_prompt(state, section, retry, difficulty),  # type: ignore
            json_mode=True,
            max_tokens=450,
            temperature=0.45,
        )
        result = parse_json(raw)
    except Exception:
        logger.exception("Question generation failed; using fallback question")
        result = {}

    question = str(
        result.get("question_text")
        or result.get("question")
        or fallback["question_text"]
    ).strip()
    if not question:
        question = fallback["question_text"]
    difficulty = str(result.get("difficulty") or fallback["difficulty"]).lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = fallback["difficulty"]
    if retry:
        difficulty = "easy"
    concept = str(
        result.get("concept_tag") or result.get("concept") or fallback["concept_tag"]
    ).strip()
    if retry:
        concept = str(state.get("current_question_concept") or concept)

    turn_number = int(state.get("turn_number") or 0) + 1
    sections_patch, used, questions_asked = _update_section_for_question(state, concept)
    section_name = str(section.get("section_name") or section.get("name") or "general")
    turn = {
        "turn_number": turn_number,
        "speaker": "bot",
        "text": question,
        "tone": "neutral",
        "section": section_name,
        "turn_type": "question",
        "type": "question",
        "difficulty": difficulty,
        "concept": concept,
        "concept_tag": concept,
        "weak_retry": retry,
        "timestamp": time.time(),
    }
    return {
        "current_question": question,
        "current_question_text": question,
        "current_question_difficulty": difficulty,
        "current_question_concept": concept,
        "turn_number": turn_number,
        "last_bot_text": question,
        "bot_reply_text": question,
        "bot_reply_type": "question",
        "questions_asked_in_section": questions_asked,
        "concepts_covered_in_section": used,
        "used_concepts": used,
        "sections": sections_patch,
        "candidate_raw_text": "",
        "response_class": None,
        "next_question_mode": "normal",
        "last_question_was_weak_retry": retry,
        "awaiting_think_decision": False,
        "think_timer_active": False,
        "skip_requested": False,
        "transcript": [turn],
        "next_node": "await_response",
    }
