"""Section transition node."""

from __future__ import annotations

import logging
import time

from src.config.settings import settings
from src.control.agents.llm import groq_complete
from src.control.agents.prompts import trim_to_2_sentences
from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)

_TRANSITION_SYSTEM = """\
You are a senior interviewer transitioning between interview sections. Generate a natural spoken transition.
Keep it to 1-2 sentences. Acknowledge the section just completed and introduce the next section warmly.
No markdown, no bullets, no evaluation.
"""


def _fallback(previous: str, next_name: str, next_skill: str | None) -> str:
    prev = previous.lower()
    nxt = next_name.lower()
    if prev == "self_intro":
        return "Thank you for that introduction. Let's move into the role-specific questions now."
    if nxt in {"behavioural", "behavioral"}:
        return "Good, let's shift gears and talk about your work style and past experiences."
    if nxt == "cultural":
        return (
            "Thanks, let's finish with a quick look at team fit and work preferences."
        )
    if next_skill:
        return f"Good, let's move on to {next_skill}."
    return "Great, let's move on to the next area of the interview."


async def section_transition(state: InterviewState) -> dict:
    sections = [dict(section) for section in state.get("sections") or []]
    current_idx = int(state.get("current_section_index") or 0)
    now = time.time()
    if current_idx < len(sections):
        sections[current_idx]["is_complete"] = True
        sections[current_idx]["time_elapsed_secs"] = int(
            now - float(state.get("section_started_at") or now)
        )

    next_idx = current_idx + 1
    if next_idx >= len(sections):
        return {
            "sections": sections,
            "current_section_index": next_idx,
            "should_close": True,
            "session_status": "COMPLETED",
            "next_node": "closing",
        }

    previous = str(state.get("current_section_name") or "")
    next_section = sections[next_idx]
    next_name = str(
        next_section.get("section_name") or next_section.get("name") or "general"
    )
    next_skill = str(next_section.get("skill") or "") or None
    try:
        text = await groq_complete(
            system=_TRANSITION_SYSTEM,
            user=f"From section: {previous}\nTo section: {next_name}\nNext skill: {next_skill or 'general'}\nGenerate the transition.",
            max_tokens=120,
            temperature=0.7,
            model=settings.GROQ_CLASSIFY_MODEL,
        )
        text = text.strip()
        if len(text) < 10:
            raise ValueError("Transition too short")
    except Exception:
        logger.exception("Transition generation failed; using fallback")
        text = _fallback(previous, next_name, next_skill)
    text = trim_to_2_sentences(text)

    turn_number = int(state.get("turn_number") or 0) + 1
    turn = {
        "turn_number": turn_number,
        "speaker": "bot",
        "text": text,
        "tone": "neutral",
        "section": next_name,
        "turn_type": "transition",
        "type": "transition",
        "from_section": previous,
        "to_section": next_name,
        "timestamp": now,
    }
    return {
        "sections": sections,
        "current_section_index": next_idx,
        "current_section_name": next_name,
        "section_started_at": now,
        "section_allocated_secs": float(next_section.get("time_budget_secs") or 60),  # type: ignore
        "current_section_time_remaining_secs": int(  # type: ignore
            next_section.get("time_budget_secs") or 60
        ),
        "questions_asked_in_section": 0,
        "concepts_covered_in_section": [],
        "used_concepts": [],
        "current_difficulty_level": 1,
        "next_question_mode": "normal",
        "last_question_was_weak_retry": False,
        "turn_number": turn_number,
        "last_bot_text": text,
        "bot_reply_text": text,
        "bot_reply_type": "transition",
        "candidate_raw_text": "",
        "response_class": None,
        "silence_state": {},
        "transcript": [turn],
        "next_node": "generate_question",
    }
