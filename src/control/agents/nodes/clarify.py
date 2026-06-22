"""Clarification handling node."""

from __future__ import annotations

import logging
import random
import time

from src.config.settings import settings
from src.control.agents.llm import groq_complete
from src.control.agents.prompts import trim_to_2_sentences
from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)

_REPHRASE_SYSTEM = """\
You are a senior interviewer. The candidate asked you to clarify or repeat your question.
Start with a brief warm acknowledgment, rephrase the question using simpler words, and keep it to 2-3 spoken sentences.
Do not evaluate the candidate. Do not use markdown.
"""

_FALLBACKS = [
    "Of course. Let me put it differently: {question}",
    "Sure thing. What I'm really asking is this: {question}",
    "Absolutely, let me rephrase that: {question}",
    "No problem. Here's another way to think about it: {question}",
]


async def handle_clarification(state: InterviewState) -> dict:
    raw = state.get("candidate_raw_text") or state.get("last_transcript") or ""
    question = (
        state.get("current_question_text")
        or state.get("current_question")
        or "Could you elaborate?"
    )
    sections = state.get("sections") or []
    idx = int(state.get("current_section_index") or 0)
    section = sections[idx] if idx < len(sections) else {}
    skill = section.get("skill") or section.get("section_name") or "general"

    try:
        response = await groq_complete(
            system=_REPHRASE_SYSTEM,
            user=f"Skill being assessed: {skill}\nOriginal question: {question}\nCandidate request: {raw}\nRephrase now.",
            max_tokens=170,
            temperature=0.3,
            model=settings.GROQ_CLASSIFY_MODEL,
        )
        response = response.strip()
        if len(response) < 10:
            raise ValueError("Rephrase too short")
    except Exception:
        logger.exception("Clarification model failed; using fallback")
        response = random.choice(_FALLBACKS).format(question=question)

    response = trim_to_2_sentences(response)
    turn_number = int(state.get("turn_number") or 0) + 1
    turn = {
        "turn_number": turn_number,
        "speaker": "bot",
        "text": response,
        "tone": "neutral",
        "section": state.get("current_section_name", "general"),
        "turn_type": "clarification",
        "type": "clarification",
        "timestamp": time.time(),
    }
    return {
        "last_bot_text": response,
        "bot_reply_text": response,
        "bot_reply_type": "clarification",
        "turn_number": turn_number,
        "candidate_raw_text": "",
        "response_class": None,
        "transcript": [turn],
        "silence_state": {},
        "next_node": "await_response",
    }
