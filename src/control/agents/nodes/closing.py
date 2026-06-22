"""Closing node for completed or terminated interviews."""

from __future__ import annotations

import logging
import time

from src.config.settings import settings
from src.control.agents.llm import groq_complete
from src.control.agents.prompts import trim_to_2_sentences
from src.control.agents.state import InterviewState
from src.control.time_manager import compute_elapsed
from src.data.repositories import interview_workflow_repository as db

logger = logging.getLogger(__name__)

_CLOSING_SYSTEM = """\
You are a senior interviewer wrapping up a live voice interview. Generate a warm, professional closing statement.
Keep it to 3-4 spoken sentences. Thank the candidate, briefly mention areas covered, and say the team will review and get back with next steps.
Do not reveal evaluation or scores. No markdown.
"""


def _covered_sections(state: InterviewState) -> list[str]:
    sections = state.get("sections") or []
    return [
        str(section.get("section_name") or section.get("name"))
        for section in sections
        if section.get("is_complete") or int(section.get("questions_asked") or 0) > 0
    ]


def _fallback_text(state: InterviewState, final_status: str) -> str:
    if final_status == "TERMINATED" and state.get("last_bot_text"):
        return str(state["last_bot_text"])
    if state.get("auto_submit_triggered"):
        return "We've reached the end of the scheduled interview time. Thank you for your responses today. Our team will review everything and get back to you with next steps."
    covered = _covered_sections(state)
    section_text = ", ".join(covered[:3]) if covered else "several areas"
    return f"That brings us to the end of the interview. We covered {section_text}, and I appreciate your time today. Our team will review everything and get back to you with next steps."


async def closing(state: InterviewState) -> dict:
    if state.get("closing_done"):
        return {"next_node": "trigger_evaluation"}

    final_status = (
        "TERMINATED" if state.get("session_status") == "TERMINATED" else "COMPLETED"
    )
    try:
        text = await groq_complete(
            system=_CLOSING_SYSTEM,
            user=(
                f"Status: {final_status}\n"
                f"Sections covered: {', '.join(_covered_sections(state)) or 'several areas'}\n"
                f"Turns: {state.get('turn_number') or 0}\n"
                "Generate the closing statement."
            ),
            max_tokens=190,
            temperature=0.7,
            model=settings.GROQ_CLASSIFY_MODEL,
        )
        text = text.strip()
        if len(text) < 20:
            raise ValueError("Closing too short")
    except Exception:
        logger.exception("Closing generation failed; using fallback")
        text = _fallback_text(state, final_status)
    text = trim_to_2_sentences(text) if final_status == "TERMINATED" else text

    existing_transcript = state.get("transcript") or []
    already_spoke_closing = (
        existing_transcript
        and existing_transcript[-1].get("speaker") == "bot"
        and existing_transcript[-1].get("turn_type") == "closing"
    )
    turn_number = int(state.get("turn_number") or 0) + (
        0 if already_spoke_closing else 1
    )
    closing_turns = (
        []
        if already_spoke_closing
        else [
            {
                "turn_number": turn_number,
                "speaker": "bot",
                "text": text,
                "tone": "neutral",
                "section": state.get("current_section_name", "closing"),
                "turn_type": "closing",
                "type": "closing",
                "timestamp": time.time(),
            }
        ]
    )
    patch = {
        "last_bot_text": text,
        "bot_reply_text": text,
        "bot_reply_type": "closing",
        "turn_number": turn_number,
        "transcript": closing_turns,
        "session_status": final_status,
        "closing_done": True,
        "should_close": True,
        "candidate_raw_text": "",
        "response_class": None,
        "total_elapsed_secs": compute_elapsed(state),
        "next_node": "trigger_evaluation",
    }
    final_state = {**state, **patch}
    final_state["transcript"] = [*(state.get("transcript") or []), *closing_turns]
    final_state["violations"] = state.get("violations") or []
    final_state["question_scores"] = state.get("question_scores") or []
    final_state["answer_evaluations"] = state.get("answer_evaluations") or []
    await db.persist_session_state(final_state)
    try:
        await db.mark_candidate_finished(state["candidate_assessment_id"])
    except Exception:
        logger.exception("Failed to mark candidate assessment finished")
    return patch
