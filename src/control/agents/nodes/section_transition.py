"""
section_transition_node — Manages transitions between interview sections.

Handles:
1. Marking the current section as complete
2. Redistributing unused time to remaining sections (proportional to priority)
3. Generating a natural-sounding transition phrase via LLM
4. Resetting per-section counters (used_concepts, consecutive scores, etc.)
5. Routing to interview_complete when all sections are done

Time management:
  - If a section finishes early, its leftover time is redistributed
    proportionally by priority_score to the remaining sections.
  - If a section runs over, the deficit is absorbed from remaining
    sections (also proportional to priority).
"""

from __future__ import annotations

import logging

from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_fixed

from src.config.settings import settings
from src.control.agents.state import InterviewState, SectionState

logger = logging.getLogger(__name__)


_TRANSITION_SYSTEM_PROMPT = """\
You are a senior technical interviewer conducting a live voice interview. \
You just finished one section and are about to start another.

Generate a natural spoken transition between sections. Rules:
- Keep it to 2 sentences maximum. This is spoken aloud via TTS.
- Briefly acknowledge the section you're leaving (1 short phrase).
- Introduce the next section warmly. Sound like a real interviewer, not a script.
- Do NOT use bullet points, markdown, or numbered lists.
- Vary your style. Don't always say "Great, let's move on."
- Examples of natural transitions:
  "Good stuff on the Python side. Let's shift gears and talk about system design."
  "Alright, I've got a good sense of your backend skills. Now I'd love to hear about how you approach problem solving."
  "Nice, thanks for walking me through that. So, switching topics a bit..."
"""


@retry(stop=stop_after_attempt(2), wait=wait_fixed(1))
async def _generate_transition(
    from_section: str,
    to_section: str,
    to_skill: str,
) -> str:
    """Generate a natural-sounding section transition via LLM."""
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    completion = await client.chat.completions.create(
        model=settings.GROQ_CLASSIFY_MODEL,  # Fast 8b model for transitions
        messages=[
            {"role": "system", "content": _TRANSITION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Transitioning from: {from_section}\n"
                    f"Moving to: {to_section} (skill area: {to_skill})\n"
                    f"Generate the transition now."
                ),
            },
        ],
        max_tokens=128,
        temperature=0.8,
    )
    return (completion.choices[0].message.content or "").strip()


def _redistribute_time(
    sections: list[SectionState],
    current_idx: int,
    leftover_secs: int,
) -> list[SectionState]:
    """
    Redistribute leftover time (positive or negative) from the current
    section to remaining incomplete sections, proportional to priority.

    Returns a new list of sections with updated time_budget_secs.
    """
    updated = list(sections)
    remaining_indices = [
        i
        for i in range(current_idx + 1, len(sections))
        if not sections[i].get("is_complete", False)
    ]

    if not remaining_indices or leftover_secs == 0:
        return updated

    # Sum of priority scores for remaining sections
    total_priority = sum(
        sections[i].get("priority_score", 5) for i in remaining_indices
    )
    if total_priority == 0:
        total_priority = len(remaining_indices)  # Equal split fallback

    for i in remaining_indices:
        priority = sections[i].get("priority_score", 5)
        fraction = priority / total_priority
        time_delta = int(leftover_secs * fraction)

        sec = sections[i]
        new_budget = max(
            60,  # Minimum 1 minute per section
            sec.get("time_budget_secs", 300) + time_delta,
        )

        updated[i] = SectionState(
            name=sec.get("name", ""),
            skill=sec.get("skill", ""),
            priority_score=sec.get("priority_score", 5),
            time_budget_secs=new_budget,
            time_elapsed_secs=sec.get("time_elapsed_secs", 0),
            questions_asked=sec.get("questions_asked", 0),
            concepts_covered=list(sec.get("concepts_covered", [])),
            is_complete=sec.get("is_complete", False),
        )

    return updated


async def section_transition_node(state: InterviewState) -> dict:
    """
    Transition to the next interview section.

    1. Marks the current section as complete.
    2. Calculates leftover time and redistributes to remaining sections.
    3. Generates a natural-sounding spoken transition.
    4. Resets per-section counters.
    5. Routes to interview_complete if all sections are exhausted.
    """
    sections = state.get("sections", [])
    current_idx = state.get("current_section_index", 0)
    next_idx = current_idx + 1
    time_remaining = state.get("current_section_time_remaining_secs", 0)

    # Check for global time exhaustion
    total_elapsed = state.get("total_elapsed_secs", 0)
    interview_plan = state.get("interview_plan", {})
    total_budget_secs = interview_plan.get("total_mins", 0) * 60

    if total_budget_secs > 0 and total_elapsed >= total_budget_secs:
        logger.info("Total interview time exceeded — routing to interview_complete")
        return {
            "session_status": "completed",
            "bot_reply_text": (
                "That covers everything I wanted to explore today. "
                "Let me wrap up the interview."
            ),
            "bot_reply_type": "transition",
        }

    # All sections done → route to interview_complete
    if next_idx >= len(sections):
        logger.info("No more sections — routing to interview_complete")
        return {
            "session_status": "completed",
            "bot_reply_text": (
                "That covers everything I wanted to explore today. "
                "Let me wrap up the interview."
            ),
            "bot_reply_type": "transition",
        }

    # Mark current section complete
    current_sec = sections[current_idx]
    completed_section: SectionState = {
        "name": current_sec.get("name", ""),
        "skill": current_sec.get("skill", ""),
        "priority_score": current_sec.get("priority_score", 5),
        "time_budget_secs": current_sec.get("time_budget_secs", 300),
        "time_elapsed_secs": (
            current_sec.get("time_budget_secs", 300) - time_remaining
        ),
        "questions_asked": current_sec.get("questions_asked", 0),
        "concepts_covered": list(current_sec.get("concepts_covered", [])),
        "is_complete": True,
    }

    # Redistribute leftover time
    updated_sections = list(sections)
    updated_sections[current_idx] = completed_section

    if time_remaining > 0:
        logger.info(
            "Section '%s' finished with %ds leftover — redistributing",
            current_sec.get("name", "?"),
            time_remaining,
        )
        updated_sections = _redistribute_time(
            updated_sections, current_idx, time_remaining
        )
    elif time_remaining < 0:
        # Section ran over — absorb deficit from remaining
        logger.info(
            "Section '%s' ran over by %ds — absorbing from remaining",
            current_sec.get("name", "?"),
            abs(time_remaining),
        )
        updated_sections = _redistribute_time(
            updated_sections, current_idx, time_remaining
        )

    next_section = updated_sections[next_idx]
    from_name = current_sec.get("name", "the previous section")
    to_name = next_section.get("name", "the next section")
    to_skill = next_section.get("skill", "general")

    reply = await _generate_transition(from_name, to_name, to_skill)
    if not reply or len(reply) < 10:
        raise ValueError("Transition text too short")

    logger.info(
        "Section transition: '%s' → '%s' (next budget: %ds)",
        from_name,
        to_name,
        next_section.get("time_budget_secs", 300),
    )

    return {
        "sections": updated_sections,
        "current_section_index": next_idx,
        "current_section_time_remaining_secs": next_section.get(
            "time_budget_secs", 300
        ),
        "current_question_difficulty": "medium",
        # Reset per-section counters
        "consecutive_strong": 0,
        "consecutive_weak": 0,
        "used_concepts": [],
        "silence_attempt": 0,
        "irrelevant_strike_count": 0,
        "bot_reply_text": reply,
        "bot_reply_type": "transition",
    }
