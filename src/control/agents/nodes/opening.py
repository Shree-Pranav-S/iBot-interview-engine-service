"""Opening/self-introduction question node."""

from __future__ import annotations

from src.control.agents.prompts import trim_to_2_sentences
from src.control.agents.state import InterviewState
from src.data.repositories import interview_workflow_repository as db


def _display_company(name: str | None) -> str:
    company = (name or "the company").strip()
    return company.title() if company.islower() else company


def _mark_question_on_section(state: InterviewState, concept: str) -> list[dict]:
    sections = [dict(section) for section in state.get("sections") or []]
    idx = int(state.get("current_section_index") or 0)
    if idx < len(sections):
        concepts = list(sections[idx].get("concepts_covered") or [])  # type: ignore
        if concept and concept not in concepts:
            concepts.append(concept)
        sections[idx]["questions_asked"] = (
            int(sections[idx].get("questions_asked") or 0) + 1  # type: ignore
        )
        sections[idx]["concepts_covered"] = concepts
    return sections


async def opening(state: InterviewState) -> dict:
    await db.mark_candidate_started(state["candidate_assessment_id"])
    company = _display_company(state.get("company_name"))
    text = trim_to_2_sentences(
        "Hi and welcome to this "
        f"{company} interview, can you start by telling me a little about "
        "yourself, including your background and what inspired you to apply "
        "for this role?"
    )
    turn_number = int(state.get("turn_number") or 0) + 1
    concept = "self_introduction"
    sections = _mark_question_on_section(state, concept)
    return {
        "session_status": "IN_PROGRESS",
        "current_question": text,
        "current_question_text": text,
        "current_question_difficulty": "easy",
        "current_question_concept": concept,
        "last_bot_text": text,
        "bot_reply_text": text,
        "bot_reply_type": "opening",
        "turn_number": turn_number,
        "questions_asked_in_section": 1,
        "concepts_covered_in_section": [concept],
        "used_concepts": [concept],
        "sections": sections,
        "candidate_raw_text": "",
        "response_class": None,
        "transcript": [
            {
                "turn_number": turn_number,
                "speaker": "bot",
                "text": text,
                "tone": "neutral",
                "section": state.get("current_section_name", "self_intro"),
                "turn_type": "question",
                "difficulty": "easy",
                "concept": concept,
            }
        ],
        "next_node": "await_response",
    }
