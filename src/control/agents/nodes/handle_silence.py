"""
handle_silence_node — Multi-attempt silence protocol.

Manages a 3-stage escalation when the candidate goes silent:

  Attempt 0 → Gentle, warm nudge. Sounds like the interviewer noticed
              a natural pause and is giving them space.
  Attempt 1 → Offer think-time. Explicitly tells the candidate it's
              okay to take a moment, and offers to rephrase.
  Attempt 2 → Zero-score this question and move on. The bot sounds
              understanding, not punitive.

The conditional edge after this node checks silence_attempt:
  - If reset to 0 → zero-score path → route to generate_question
  - Otherwise → route back to await_response
"""

from __future__ import annotations

import logging
import random

from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)


# ── Natural-sounding nudge variations ─────────────────────────────────────────

_GENTLE_NUDGES = [
    "Take your time — there's no rush. Whenever you're ready.",
    "I know that can be a tricky one. Take a moment to think it through.",
    "No pressure — just share whatever comes to mind.",
    "It's totally fine to think for a bit. I'm right here when you're ready.",
    "Take a breath and walk me through your thinking whenever you're ready.",
]

_THINK_OFFERS = [
    (
        "Still thinking? That's completely fine. "
        "Would it help if I rephrased the question a bit?"
    ),
    (
        "No worries at all — sometimes these things take a second to frame. "
        "If you'd like, I can come at it from a different angle."
    ),
    (
        "Take all the time you need. If you're unsure where to start, "
        "even a partial answer or a related experience would be great."
    ),
    (
        "I'll give you another moment. If this isn't your area, "
        "it's perfectly okay to say so and we can move on."
    ),
]

_MOVE_ON_PHRASES = [
    (
        "That's alright — not every question clicks, and that's completely "
        "normal. Let's move on to something else."
    ),
    ("No problem at all. Let's switch gears and talk about something different."),
    (
        "That's fine — we'll come back to this area if we have time. "
        "Let me ask you about something else."
    ),
    ("Totally understandable. Let's try a different topic."),
]


async def handle_silence_node(state: InterviewState) -> dict:
    """
    Handle silence from the candidate using a 3-stage escalation protocol.

    Each stage sounds natural and empathetic — never robotic or punitive.
    """
    silence_attempt = state.get("silence_attempt", 0)
    sections = state.get("sections", [])
    section_idx = state.get("current_section_index", 0)
    turn_number = state.get("turn_number", 0)

    current_section = sections[section_idx] if section_idx < len(sections) else None
    section_name = current_section["name"] if current_section else "unknown"
    skill = current_section["skill"] if current_section else "general"

    if silence_attempt == 0:
        # Stage 1: Gentle nudge
        reply = random.choice(_GENTLE_NUDGES)
        new_attempt = 1
        logger.info("Silence protocol: gentle nudge (attempt 0)")

        return {
            "silence_attempt": new_attempt,
            "bot_reply_text": reply,
            "bot_reply_type": "nudge",
        }

    elif silence_attempt == 1:
        # Stage 2: Offer think time / rephrase offer
        reply = random.choice(_THINK_OFFERS)
        new_attempt = 2
        logger.info("Silence protocol: think offer (attempt 1)")

        return {
            "silence_attempt": new_attempt,
            "think_timer_active": True,
            "bot_reply_text": reply,
            "bot_reply_type": "nudge",
        }

    else:
        # Stage 3: Zero-score and move on
        reply = random.choice(_MOVE_ON_PHRASES)
        logger.info("Silence protocol: zero-score, moving on (attempt 2)")

        # Build a zero-score evaluation record
        question = state.get("current_question_text", "")
        eval_record = {
            "turn_number": turn_number,
            "section": section_name,
            "skill": skill,
            "question": question,
            "answer": "",
            "quality": "non_answer",
            "score": 0,
            "signals_present": [],
            "signals_missing": [],
            "is_substantial": False,
            "key_concept_demonstrated": "",
            "one_line_feedback": "No response — silence timeout after 3 attempts.",
        }

        # Record transcript turn for the silence
        turn_record = {
            "turn_number": turn_number,
            "speaker": "candidate",
            "text": "[No response — silence]",
            "section": section_name,
            "type": "silence",
            "evaluation": {
                "quality": "non_answer",
                "score": 0,
            },
        }

        return {
            "silence_attempt": 0,  # Reset for next question
            "think_timer_active": False,
            "bot_reply_text": reply,
            "bot_reply_type": "nudge",
            "answer_evaluations": (state.get("answer_evaluations", []) + [eval_record]),
            "transcript_turns": (state.get("transcript_turns", []) + [turn_record]),
        }
