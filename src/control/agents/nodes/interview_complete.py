"""
interview_complete_node — Delivers the closing statement.

Generates a natural-sounding closing monologue via LLM that:
- Thanks the candidate warmly
- Gives a brief summary of what was covered
- Sets expectations for next steps
- Sounds like a real interviewer wrapping up, not a script
"""

from __future__ import annotations

import logging

from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_fixed

from src.config.settings import settings
from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)


_CLOSING_SYSTEM_PROMPT = """\
You are a senior technical interviewer wrapping up a live voice interview. \
Generate a warm, professional closing statement.

Rules:
- Keep it to 3-4 sentences. This is spoken aloud via TTS.
- Thank the candidate for their time.
- Briefly mention what areas were covered (you'll be given section names).
- Set expectations: "our team will review and get back to you with next steps."
- Sound genuine and warm — not formulaic.
- Do NOT use bullet points or markdown.
- Do NOT give any hint about how well they did.
"""


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
async def _generate_closing(
    sections_covered: list[str],
    turn_count: int,
    was_terminated: bool,
) -> str:
    """Generate a natural closing monologue."""
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)

    if was_terminated:
        context = (
            "The interview was terminated early due to off-topic responses. "
            "Be brief and professional — don't blame the candidate."
        )
    else:
        context = (
            f"The interview covered {len(sections_covered)} sections: "
            f"{', '.join(sections_covered)}. "
            f"Total of about {turn_count} exchanges."
        )

    completion = await client.chat.completions.create(
        model=settings.GROQ_CLASSIFY_MODEL,  # Fast 8b model
        messages=[
            {"role": "system", "content": _CLOSING_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        max_tokens=192,
        temperature=0.8,
    )
    return (completion.choices[0].message.content or "").strip()


async def interview_complete_node(state: InterviewState) -> dict:
    """
    Deliver the closing statement and mark the interview as complete.

    Uses LLM for natural closing with template fallback.
    """
    sections = state.get("sections", [])
    turn_number = state.get("turn_number", 0)
    status = state.get("session_status", "completed")
    was_terminated = status == "terminated"

    # Gather section names that were covered
    sections_covered = [
        s["name"]
        for s in sections
        if s.get("is_complete") or s.get("questions_asked", 0) > 0
    ]

    try:
        reply = await _generate_closing(sections_covered, turn_number, was_terminated)
        if not reply or len(reply) < 20:
            raise ValueError("Closing text too short")
    except Exception:
        logger.warning("LLM closing generation failed, using template")
        if was_terminated:
            reply = (
                "Thank you for your time today. Unfortunately, we weren't "
                "able to cover everything we planned, but we appreciate you "
                "joining us. Our team will review the session. Best of luck."
            )
        else:
            section_list = (
                ", ".join(sections_covered[:3]) if sections_covered else "several areas"
            )
            reply = (
                f"That brings us to the end of the interview. We covered "
                f"{section_list}, and I really appreciate your thoughtful "
                f"responses throughout. Our team will review everything "
                f"and get back to you with next steps. Thank you so much "
                f"for your time, and best of luck!"
            )

    logger.info(
        "Interview complete: assessment=%s turns=%d status=%s",
        state.get("candidate_assessment_id", "unknown"),
        turn_number,
        status,
    )

    return {
        "bot_reply_text": reply,
        "bot_reply_type": "closing",
        "session_status": "completed" if not was_terminated else "terminated",
    }
