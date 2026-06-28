"""Deterministic section context and technical difficulty strategy."""

from __future__ import annotations

from typing import Any

from src.control.agents.state import Difficulty, InterviewState

DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")
ROLE_INITIAL_DIFFICULTY: dict[str, Difficulty] = {
    "junior level": "easy",
    "mid-level": "medium",
    "senior level": "hard",
}


def resume_has_skill(state: InterviewState, skill: str | None) -> bool:
    """Use tolerant case-insensitive matching for normalized resume skills."""

    target = str(skill or "").strip().casefold()
    if not target:
        return False
    for raw in (state.get("resume_context") or {}).get("skills", []):
        candidate = str(raw).strip().casefold()
        if candidate == target:
            return True
        if (
            candidate
            and min(len(candidate), len(target)) >= 3
            and (candidate in target or target in candidate)
        ):
            return True
    return False


def target_section(state: InterviewState) -> tuple[int, dict[str, Any]]:
    """Return the next section selected by normal or forced timing control."""

    sections = list(state.get("runtime_sections") or [])
    if not sections:
        raise RuntimeError("No runtime interview sections are available")

    pending_index = state.get("pending_section_index")
    if pending_index is not None:
        bounded_pending = min(max(0, int(pending_index)), len(sections) - 1)
        return bounded_pending, sections[bounded_pending]

    current_index = int(state.get("current_section_index") or 0)
    current_kind = state.get("current_section_kind")
    if current_kind == "self_intro":
        for index in range(current_index + 1, len(sections)):
            section = sections[index]
            if section.get("section_kind") in {
                "technical",
                "behavioural_cultural",
            }:
                return index, section

    bounded_index = min(max(0, current_index), len(sections) - 1)
    return bounded_index, sections[bounded_index]


def _one_step(current: Difficulty, direction: int) -> Difficulty:
    index = DIFFICULTIES.index(current)
    return DIFFICULTIES[min(2, max(0, index + direction))]


def determine_question_difficulty(
    state: InterviewState,
    *,
    skill: str,
    entering_new_section: bool,
) -> tuple[Difficulty, bool]:
    """Return target difficulty and whether this is the single weak-answer probe."""

    resume_floor = resume_has_skill(state, skill)
    initial = ROLE_INITIAL_DIFFICULTY.get(
        str(state.get("inferred_difficulty") or "").lower(),
        "medium",
    )
    if entering_new_section or state.get("current_question_difficulty") is None:
        target = initial
        probe_deeper = False
    else:
        current = state.get("current_question_difficulty") or initial
        target = current
        probe_deeper = False
        evaluation = (
            state.get("latest_evaluation")
            if (
                state.get("last_response_type") == "answer"
                and state.get("last_response_substantial")
            )
            else None
        )
        strength = (
            str(evaluation.get("strength") or "")
            if isinstance(evaluation, dict)
            else ""
        )
        streaks = (state.get("skill_evaluation_streaks") or {}).get(
            skill.casefold(), {}
        )

        if strength == "weak":
            target = _one_step(current, -1)
            probe_deeper = int(streaks.get("weak") or 0) == 1
        elif strength == "adequate":
            if int(streaks.get("adequate") or 0) >= 2:
                target = _one_step(current, 1)
        elif strength == "strong":
            target = _one_step(current, 1)

    if resume_floor and target == "easy":
        target = "medium"
    return target, probe_deeper
