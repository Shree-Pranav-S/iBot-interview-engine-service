"""
handle_irrelevant_node — Handles off-topic or irrelevant responses.

Implements a 3-strike system with escalating firmness:

  Strike 1 → Gentle redirect. Sounds like the interviewer is casually
             steering the conversation back. Doesn't feel like a warning.
  Strike 2 → Firmer redirect. Makes it clear the answer needs to be
             on-topic, but stays professional and empathetic.
  Strike 3 → Terminates the interview gracefully. The bot sounds
             understanding, not accusatory.

Each redirect uses an LLM call for natural phrasing when possible,
with template fallbacks for reliability.
"""

from __future__ import annotations

import logging
import random

from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_fixed

from src.config.settings import settings
from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)


# ── Redirect templates (used as fallbacks) ────────────────────────────────────

_GENTLE_REDIRECTS = [
    (
        "I appreciate you sharing that. Let's circle back to the question "
        "though — {question_summary}"
    ),
    (
        "That's interesting context. To stay on track with our interview "
        "though, could you address — {question_summary}"
    ),
    (
        "Sure, I hear you. But let's focus on the interview question — "
        "{question_summary}"
    ),
]

_FIRM_REDIRECTS = [
    (
        "I understand, but we do need to stay focused on the interview "
        "topics. Let me ask the question again — {question_summary}"
    ),
    (
        "I want to make sure we cover what we need to in our time together. "
        "Could you try to answer this one directly? {question_summary}"
    ),
    (
        "We're running on limited time, so let's make sure we stay on topic. "
        "The question was — {question_summary}"
    ),
]

_TERMINATE_MESSAGES = [
    (
        "I appreciate your time today. Unfortunately, we haven't been "
        "able to stay focused on the interview questions, so I think "
        "it's best we wrap up here. Thank you for your participation, "
        "and best of luck."
    ),
    (
        "Thank you for joining today. Since we haven't been able to "
        "cover the interview material, I'm going to go ahead and "
        "conclude the session. We appreciate your time."
    ),
]

_REDIRECT_SYSTEM_PROMPT = """\
You are a professional technical interviewer. The candidate just gave an \
off-topic or irrelevant response. You need to gently redirect them back \
to the interview question.

Rules:
- Be polite but firm. Do NOT sound annoyed or punitive.
- Briefly acknowledge what they said (1 short phrase), then redirect.
- Restate the core of the interview question in your own words.
- Keep it to 2 sentences maximum. This is spoken aloud.
- Do NOT use bullet points or markdown.
- Do NOT evaluate or score their response.
"""


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
async def _generate_redirect(
    question: str,
    transcript: str,
    skill: str,
    strike: int,
) -> str:
    """Generate a natural redirect using LLM."""
    firmness = "gently" if strike == 1 else "firmly but professionally"
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    completion = await client.chat.completions.create(
        model=settings.GROQ_CLASSIFY_MODEL,  # Use fast 8b model
        messages=[
            {"role": "system", "content": _REDIRECT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"The interview question was about: {skill}\n"
                    f"Question: {question}\n"
                    f"Candidate said: {transcript[:200]}\n"
                    f"This is strike {strike}/3. Redirect them {firmness}."
                ),
            },
        ],
        max_tokens=128,
        temperature=0.7,
    )
    return (completion.choices[0].message.content or "").strip()


async def handle_irrelevant_node(state: InterviewState) -> dict:
    """
    Handle an irrelevant/off-topic response with escalating firmness.

    Strike 1: Gentle redirect
    Strike 2: Firm redirect
    Strike 3: Terminate interview
    """
    strike_count = state.get("irrelevant_strike_count", 0) + 1
    question = state.get("current_question_text", "")
    transcript = state.get("last_transcript", "")
    sections = state.get("sections", [])
    section_idx = state.get("current_section_index", 0)
    turn_number = state.get("turn_number", 0)

    current_section = sections[section_idx] if section_idx < len(sections) else None
    section_name = current_section["name"] if current_section else "unknown"
    skill = current_section["skill"] if current_section else "general"

    # Strike 3: Terminate
    if strike_count >= 3:
        reply = random.choice(_TERMINATE_MESSAGES)
        logger.warning(
            "Irrelevant strike limit reached (%d): terminating interview",
            strike_count,
        )

        # Record final evaluation
        eval_record = {
            "turn_number": turn_number,
            "section": section_name,
            "skill": skill,
            "question": question,
            "answer": transcript,
            "quality": "non_answer",
            "score": 0,
            "signals_present": [],
            "signals_missing": [],
            "is_substantial": False,
            "key_concept_demonstrated": "",
            "one_line_feedback": (
                f"Irrelevant response — strike {strike_count}/3. Interview terminated."
            ),
        }

        turn_record = {
            "turn_number": turn_number,
            "speaker": "candidate",
            "text": transcript,
            "section": section_name,
            "type": "irrelevant",
            "evaluation": {"quality": "non_answer", "score": 0},
        }

        return {
            "irrelevant_strike_count": strike_count,
            "bot_reply_text": reply,
            "bot_reply_type": "closing",
            "session_status": "terminated",
            "answer_evaluations": (state.get("answer_evaluations", []) + [eval_record]),
            "transcript_turns": (state.get("transcript_turns", []) + [turn_record]),
        }

    # Strike 1 or 2: Redirect
    question_summary = question[:150] if question else "the interview question"

    # Try LLM-generated redirect for naturalness
    try:
        reply = await _generate_redirect(question, transcript, skill, strike_count)
        if not reply or len(reply) < 10:
            raise ValueError("LLM response too short")
    except Exception:
        logger.warning("LLM redirect failed, using template (strike %d)", strike_count)
        if strike_count == 1:
            template = random.choice(_GENTLE_REDIRECTS)
        else:
            template = random.choice(_FIRM_REDIRECTS)
        reply = template.format(question_summary=question_summary)

    logger.info("Irrelevant response: strike %d/3", strike_count)

    # Record evaluation
    eval_record = {
        "turn_number": turn_number,
        "section": section_name,
        "skill": skill,
        "question": question,
        "answer": transcript,
        "quality": "non_answer",
        "score": 0,
        "signals_present": [],
        "signals_missing": [],
        "is_substantial": False,
        "key_concept_demonstrated": "",
        "one_line_feedback": (f"Off-topic response — strike {strike_count}/3."),
    }

    turn_record = {
        "turn_number": turn_number,
        "speaker": "candidate",
        "text": transcript,
        "section": section_name,
        "type": "irrelevant",
        "evaluation": {"quality": "non_answer", "score": 0},
    }

    return {
        "irrelevant_strike_count": strike_count,
        "bot_reply_text": reply,
        "bot_reply_type": "nudge",
        "answer_evaluations": (state.get("answer_evaluations", []) + [eval_record]),
        "transcript_turns": (state.get("transcript_turns", []) + [turn_record]),
    }
