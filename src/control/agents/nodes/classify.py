"""Candidate response classification node."""

from __future__ import annotations

import logging
import time

from src.config.settings import settings
from src.control.agents.llm import groq_complete
from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)

SKIP_PHRASES = ("skip", "next question", "move on", "pass this", "i pass")
THINK_PHRASES = (
    "yes",
    "yeah",
    "sure",
    "please",
    "moment",
    "think",
    "need time",
    "give me",
    "one second",
    "few seconds",
)
NO_ANSWER_PHRASES = {"no", "nope", "not really", "i don't know", "i dont know"}

_CLASSIFY_SYSTEM = """\
You are an expert technical interviewer assistant. Classify the candidate's response.

Categories:
- clarification: asking to repeat, rephrase, explain, clarify, or asking about the question/timer/audio.
- irrelevant: completely off-topic or unrelated to the current interview question.
- answer: any attempt to answer, even if short, weak, partially wrong, or uncertain.

Return ONLY one lowercase word: clarification, irrelevant, or answer.
"""

_CLASSIFY_USER = """\
Current section: {section}
Skill: {skill}
Question asked: {question}
Candidate transcript: "{transcript}"

Classification:"""


def _is_think_request(text: str) -> bool:
    lower = text.lower().strip()
    return len(lower.split()) <= 8 and any(phrase in lower for phrase in THINK_PHRASES)


def _current_section(state: InterviewState) -> dict:
    sections = state.get("sections") or []
    idx = int(state.get("current_section_index") or 0)
    return sections[idx] if idx < len(sections) else {}  # type: ignore


async def _llm_classify(state: InterviewState, text: str) -> str:
    section = _current_section(state)
    try:
        raw = await groq_complete(
            system=_CLASSIFY_SYSTEM,
            user=_CLASSIFY_USER.format(
                section=section.get("section_name") or section.get("name") or "general",
                skill=section.get("skill") or "general",
                question=state.get("current_question_text")
                or state.get("current_question")
                or "",
                transcript=text,
            ),
            max_tokens=settings.GROQ_CLASSIFY_MAX_TOKENS,
            temperature=settings.GROQ_CLASSIFY_TEMPERATURE,
            model=settings.GROQ_CLASSIFY_MODEL,
        )
        cleaned = raw.strip().strip("'\"`.,").lower()
        if cleaned in {"clarification", "irrelevant", "answer"}:
            return cleaned
        if "clarification" in cleaned:
            return "clarification"
        if "irrelevant" in cleaned:
            return "irrelevant"
    except Exception:
        logger.exception("Classification LLM call failed; defaulting to answer")
    return "answer"


async def classify_response(state: InterviewState) -> dict:
    text = " ".join((state.get("candidate_raw_text") or "").split())
    lower = text.lower().strip()

    if text == "__TIME_UP__":
        classification = "time_up"
    elif not text or text == "__SILENCE__":
        classification = "silence"
    elif bool(state.get("awaiting_think_decision")) and _is_think_request(text):
        classification = "think_request"
    elif bool(state.get("awaiting_think_decision")) and lower in NO_ANSWER_PHRASES:
        classification = "silence"
    elif any(phrase in lower for phrase in SKIP_PHRASES):
        classification = "skip"
    else:
        classification = await _llm_classify(state, text)

    turn_number = int(state.get("turn_number") or 0) + 1
    candidate_text = "" if text in {"__SILENCE__", "__TIME_UP__"} else text
    turn = {
        "turn_number": turn_number,
        "speaker": "candidate",
        "text": candidate_text,
        "tone": "neutral",
        "section": state.get("current_section_name", "general"),
        "turn_type": classification,
        "type": classification,
        "response_classification": classification,
        "stt_confidence": state.get("candidate_stt_confidence"),
        "question": state.get("current_question_text")
        or state.get("current_question", ""),
        "timestamp": time.time(),
    }

    logger.info("Candidate response classified as %s", classification)
    return {
        "response_class": classification,
        "last_response_classification": classification,
        "last_transcript": candidate_text,
        "turn_number": turn_number,
        "bot_reply_text": "",
        "bot_reply_type": "",
        "transcript": [turn],
    }
