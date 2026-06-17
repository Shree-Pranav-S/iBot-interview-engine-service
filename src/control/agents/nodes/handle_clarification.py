"""
handle_clarification_node — Rephrases the current question.

When the candidate asks for clarification, the bot rephrases the question
using an LLM call for natural, contextual rephrasing. Falls back to
template-based rephrasing if the LLM call fails.

The bot should sound like an interviewer who naturally clarifies — not
like it's reading from a script.
"""

from __future__ import annotations

import logging
import random

from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_fixed

from src.config.settings import settings
from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)


_REPHRASE_SYSTEM_PROMPT = """\
You are a senior technical interviewer. The candidate just asked you to \
clarify or repeat your question. Rephrase the question in a different way.

Rules:
- Start with a brief, warm acknowledgment (e.g., "Of course!", "Sure thing.", \
"Absolutely, let me put it differently.").
- Rephrase the question using different words and a simpler structure.
- If the original question was complex, break it down or give a hint about \
what kind of answer you're looking for.
- Keep it concise — 2-3 sentences max. This is spoken aloud.
- Do NOT use bullet points or markdown.
- Do NOT evaluate or comment on the candidate's request to clarify.
- Sound natural and conversational.
"""


_FALLBACK_TEMPLATES = [
    "Of course! Let me put it differently — {question}",
    "Sure thing. What I'm really asking is — {question}",
    "Absolutely. Let me rephrase that — {question}",
    "No problem! Here's another way to think about it — {question}",
    "Good question. Let me come at it from a different angle — {question}",
]


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
async def _rephrase_with_llm(question: str, skill: str) -> str:
    """Rephrase the interview question using LLM for natural rephrasing."""
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    completion = await client.chat.completions.create(
        model=settings.GROQ_CLASSIFY_MODEL,  # Fast 8b model
        messages=[
            {"role": "system", "content": _REPHRASE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Skill being assessed: {skill}\n"
                    f"Original question: {question}\n"
                    f"Rephrase this question now."
                ),
            },
        ],
        max_tokens=192,
        temperature=0.3,
    )
    return (completion.choices[0].message.content or "").strip()


async def handle_clarification_node(state: InterviewState) -> dict:
    """
    Rephrase the current question when the candidate asks for clarification.

    Uses LLM for natural rephrasing with template fallback.
    """
    question = state.get("current_question_text", "Could you elaborate?")
    sections = state.get("sections", [])
    section_idx = state.get("current_section_index", 0)

    current_section = sections[section_idx] if section_idx < len(sections) else None
    skill = current_section["skill"] if current_section else "general"

    # Try LLM rephrasing
    try:
        reply = await _rephrase_with_llm(question, skill)
        if not reply or len(reply) < 10:
            raise ValueError("LLM rephrase too short")
    except Exception:
        logger.warning("LLM rephrase failed, using template")
        template = random.choice(_FALLBACK_TEMPLATES)
        reply = template.format(question=question)

    logger.info("Clarification handled: question rephrased")

    return {
        "bot_reply_text": reply,
        "bot_reply_type": "clarification",
    }
