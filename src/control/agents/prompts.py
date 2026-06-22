"""Prompt templates for the live interview graph."""

from __future__ import annotations

import re

from src.control.agents.state import InterviewState
from src.control.session_loader import extract_resume_skills
from src.control.time_manager import (
    compute_section_time_remaining,
    difficulty_label,
    get_time_pressure_hint,
)

EVALUATION_SYSTEM = """\
You are an expert technical interview evaluator. Evaluate the candidate's answer against the current question, skill, section, and priority.
Return ONLY valid JSON with no markdown or code fences.
Be fair: short/partial answers can still be weak or adequate if they attempt the question, but silence/refusal is non_answer.
"""
QUESTION_GEN_SYSTEM = (
    "You are a senior interviewer conducting a structured voice interview. "
    "Generate one clear conversational question. Return valid JSON only."
)

CLARIFICATION_SYSTEM = (
    "You are a professional AI interviewer. Handle the candidate's meta-question "
    "briefly, then re-ask the original question clearly."
)


def trim_to_2_sentences(text: str) -> str:
    clean = " ".join((text or "").split())
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    if len(sentences) > 2:
        return " ".join(sentences[:2])
    return clean


def _current_section(state: InterviewState) -> dict:
    sections = state.get("sections") or []
    idx = int(state.get("current_section_index") or 0)
    if idx < len(sections):
        return sections[idx]  # type: ignore
    return {}


def _resume_summary_for_skill(state: InterviewState, skill: str | None) -> str:
    resume = state.get("resume_parsed") or state.get("resume_context") or {}
    skills = extract_resume_skills(resume)
    experience = resume.get("experience_years", "unknown")
    if skill and skills:
        matching = [item for item in skills if skill.lower() in item.lower()]
        if matching:
            return f"Experience years: {experience}; matching resume skills: {', '.join(matching[:8])}"
    return f"Experience years: {experience}; resume skills: {', '.join(skills[:12]) or 'not available'}"


def _is_technical_section(skill: str, section_name: str) -> bool:
    non_tech = {
        "communication",
        "behavioural",
        "behavioral",
        "closing",
        "general",
        "cultural",
        "culture",
        "self introduction",
        "intro",
        "introduction",
    }
    skill_lower = skill.lower()
    name_lower = section_name.lower()
    return not any(item in skill_lower or item in name_lower for item in non_tech)


def build_evaluation_prompt(state: InterviewState, answer_text: str) -> str:
    section = _current_section(state)
    section_name = str(section.get("section_name") or section.get("name") or "general")
    skill = str(section.get("skill") or section_name)
    priority = section.get("priority_score") or 5
    difficulty = state.get("current_question_difficulty", "medium")

    if _is_technical_section(skill, section_name):
        difficulty_context = f"- Question difficulty: {difficulty}"
        scoring = """For EASY questions (maximum useful score is 6):
- 1-2: weak, superficial or missing key points
- 3-4: adequate basic understanding
- 5-6: strong for this difficulty, clear example or explanation

For MEDIUM questions (maximum useful score is 8):
- 1-3: weak, misses important concepts
- 4-6: adequate, covers the main idea
- 7-8: strong, includes depth, examples, or tradeoffs

For HARD questions (full 0-10 range):
- 1-3: weak, superficial or incorrect
- 4-6: adequate understanding
- 7-8: strong, handles tradeoffs and edge cases
- 9-10: exceptional, precise and production-aware"""
    else:
        difficulty_context = ""
        scoring = """Standard scoring guide:
- 0: no relevant answer, silence, or refusal
- 1-3: weak, vague, incomplete, or misses the prompt
- 4-6: adequate, answers clearly with some useful substance
- 7-8: strong, detailed and well-structured
- 9-10: exceptional alignment, specificity, and communication"""

    return f"""
You are evaluating a live interview answer.

Context:
- Company: {state.get("company_name", "the company")}
- Role: {state.get("role_name", "the role")}
- Section: {section_name}
- Skill or signal assessed: {skill}
- Priority: {priority}/10
- Question asked: {state.get("current_question_text") or state.get("current_question", "")}
{difficulty_context}
- Candidate answer: {answer_text}
- Previously covered concepts: {state.get("used_concepts") or []}
- Resume context: {_resume_summary_for_skill(state, skill)}

Evaluate this answer. Return ONLY valid JSON:
{{
  "quality": "strong|adequate|weak|non_answer",
  "score": <0-10>,
  "signals_present": ["signal1", "signal2"],
  "signals_missing": ["signal1"],
  "is_substantial": true|false,
  "key_concept_demonstrated": "concept or empty string",
  "one_line_feedback": "brief internal note about answer quality",
  "reasoning": "brief reason for the score"
}}

Scoring guide:
{scoring}

Mark is_substantial false if the answer is too brief, vague, or generic to properly evaluate the skill.
"""


def build_generate_question_prompt(state: InterviewState) -> str:
    section = _current_section(state)
    section_name = str(section.get("section_name") or section.get("name") or "general")
    skill = section.get("skill") or section_name
    priority = section.get("priority_score") or 5
    remaining = compute_section_time_remaining(state)
    time_hint = get_time_pressure_hint(remaining)
    used = state.get("used_concepts") or state.get("concepts_covered_in_section") or []
    last_eval = state.get("last_evaluation") or {}
    missing = (
        last_eval.get("signals_missing") or last_eval.get("signals_missing", []) or []
    )
    quality = last_eval.get("quality") or "not_available"
    difficulty = difficulty_label(int(state.get("current_difficulty_level") or 2))
    retry_mode = state.get("next_question_mode") == "easier_same_concept"
    retry_concept = (
        state.get("current_question_concept")
        or last_eval.get("key_concept_demonstrated")
        or skill
    )

    if section_name == "self_intro":
        return (
            "Generate one warm self-introduction question asking the candidate to "
            "briefly summarize their background and motivation for the role. "
            'Return JSON: {"question_text": "...", "concept_tag": "self_introduction", "difficulty": "easy"}'
        )

    return f"""
You are a senior {state.get("role_name", "role")} interviewer.

Section: {section_name}
Skill or signal: {skill}
Priority: {priority}/10
Time remaining in section: {int(remaining)} seconds. {time_hint}
Resume context: {_resume_summary_for_skill(state, str(skill))}
Interview plan context: {state.get("interview_plan") or {}}

Previous question: {state.get("current_question_text") or state.get("current_question", "")}
Previous answer summary: {state.get("last_transcript", "")[:900]}
Previous answer quality: {quality}
Signals missing: {missing}

Concepts already covered. Do not repeat unless weak-retry mode is true:
{used}

Difficulty level: {difficulty}
Weak-retry mode: {retry_mode}
Weak-retry concept: {retry_concept}

Generate ONE question. Rules:
- If time_remaining < 60 seconds, ask a quick wrap-up question, not a deep dive.
- If weak-retry mode is true, ask one easier question on the weak-retry concept, then do not keep repeating it.
- If signals_missing is non-empty and weak-retry mode is false, probe one missing signal without repeating a used concept.
- Match difficulty: easy=definition/example, medium=design/tradeoff, hard=edge cases/failure modes.
- Be conversational and concise for voice delivery.

Return JSON only:
{{"question_text": "...", "concept_tag": "...", "difficulty": "easy|medium|hard"}}
"""
