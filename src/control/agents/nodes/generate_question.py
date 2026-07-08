"""Strict technical and behavioural/cultural question generation."""

from __future__ import annotations

from typing import Any

from src.control.agents.state import Difficulty, InterviewState
from src.control.agents.utils.apply_question import apply_resolved_question
from src.control.agents.utils.generate_question import (
    _generate_behavioural,
    _generate_technical,
)
from src.control.agents.utils.question_strategy import (
    determine_question_difficulty,
    target_section,
)


async def generate_next_question(state: InterviewState) -> dict[str, Any]:
    """
    Generate the next question for the selected timed section.

    This node determines the active section (technical or behavioural), chooses
    the appropriate difficulty and follow-up strategy, generates the question,
    and updates time budgets and history.

    Args:
        state: The current interview state.

    Returns:
        State updates containing the generated question, time tracking updates,
        and the routing key `next_action`.
    """

    section_index, section = target_section(state)
    section_kind = str(section.get("section_kind") or "technical")
    section_name = str(section.get("section_name") or "technical")
    expected_signals = list(section.get("expected_signals") or [])
    entering_new_section = (
        section_index != int(state.get("current_section_index") or 0)
        or state.get("pending_section_index") is not None
    )
    generation_state: InterviewState = {
        **state,
        "suppress_previous_context_for_next_question": (
            bool(state.get("suppress_previous_context_for_next_question"))
            or entering_new_section
        ),
    }
    preface = str(state.get("response_preface_text") or "").strip()
    if section_kind == "technical":
        skill = str(section.get("skill") or section_name)
        current_skill: str | None = skill
        difficulty, probe_deeper = determine_question_difficulty(
            state,
            skill=skill,
            entering_new_section=entering_new_section,
        )
        generated = await _generate_technical(
            generation_state,
            skill=skill,
            target_difficulty=difficulty,
            probe_deeper=probe_deeper,
        )
        acknowledgement = preface or generated.acknowledgement
        question_text = generated.question_text
        topic = generated.topic
        current_difficulty: Difficulty | None = difficulty
    else:
        current_skill = None
        current_difficulty = None
        probe_deeper = False
        generated_behavioural = await _generate_behavioural(
            generation_state,
            expected_signals=expected_signals,
        )
        acknowledgement = preface or generated_behavioural.acknowledgement
        question_text = generated_behavioural.question_text
        topic = generated_behavioural.signal_focus

    return apply_resolved_question(
        state,
        acknowledgement=acknowledgement,
        question_text=question_text,
        topic=topic,
        current_skill=current_skill,
        current_difficulty=current_difficulty,
        probe_deeper=probe_deeper,
    )
