"""
deliver_opening_node — Delivers a static, randomized opening monologue.

Produces the bot's greeting, welcomes the candidate, and
asks the candidate for a self-introduction.
"""

from __future__ import annotations

import logging
import random

from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)

# Five warm, concise, 2-sentence static openings to prevent repetition.
_OPENING_TEMPLATES = [
    "Hello and welcome to your interview for the {role_title} role! To start off, could you please give a brief introduction of yourself and walk me through your background?",
    "Hi there, thank you for joining us today for the {role_title} interview. Let's start with a quick introduction — could you tell me a bit about yourself and your professional background?",
    "Welcome, we're excited to talk to you today about the {role_title} position. To kick things off, could you please introduce yourself and walk me through your background?",
    "Hello, it's great to have you here for the {role_title} assessment. Let's begin with a short self-introduction and a walkthrough of your background.",
    "Hi, welcome to your {role_title} interview today. To get started, please introduce yourself and briefly walk me through your background.",
]


async def deliver_opening_node(state: InterviewState) -> dict:
    """
    Randomly select one of the static opening templates to greet and welcome the candidate.
    """
    plan = state.get("interview_plan", {})
    role_title = plan.get("role_title", "Software Engineer")
    sections = state.get("sections", [])

    template = random.choice(_OPENING_TEMPLATES)
    opening_text = template.format(role_title=role_title)

    logger.info(
        "Static opening monologue selected (%d chars) for assessment=%s",
        len(opening_text),
        state.get("candidate_assessment_id", "unknown"),
    )

    # Record this as turn 0 (bot opening)
    turn_record = {
        "turn_number": 0,
        "speaker": "bot",
        "text": opening_text,
        "section": sections[0]["name"] if sections else "opening",
        "type": "opening",
    }

    return {
        "bot_reply_text": opening_text,
        "bot_reply_type": "opening",
        "current_question_text": (
            "Please introduce yourself and walk me through your background."
        ),
        "transcript_turns": state.get("transcript_turns", []) + [turn_record],
    }
