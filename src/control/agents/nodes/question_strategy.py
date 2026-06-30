"""Deterministic section context and technical difficulty strategy."""

from __future__ import annotations

import re
from typing import Any

from src.control.agents.state import Difficulty, InterviewState

DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")
ROLE_INITIAL_DIFFICULTY: dict[str, Difficulty] = {
    "junior level": "easy",
    "mid-level": "medium",
    "senior level": "hard",
}

_PROJECT_VERBS = re.compile(
    r"\b(built|developed|designed|implemented|deployed|migrated|led|owned|"
    r"architected|maintained|scaled|refactored|shipped|launched)\b",
    re.IGNORECASE,
)
_CONCRETE_DETAIL_MARKERS = re.compile(
    r"\b(\d+\s*(years?|months?|weeks?|days?|%|users?|requests?|ms|gb|tb)|"
    r"team of|at\s+\w+|production|microservice|api|database|pipeline)\b",
    re.IGNORECASE,
)


def _skills_match(resume_skill: str, plan_skill: str) -> bool:
    """Tolerant case-insensitive skill overlap."""
    candidate = resume_skill.strip().casefold()
    target = plan_skill.strip().casefold()
    if not candidate or not target:
        return False
    if candidate == target:
        return True
    return min(len(candidate), len(target)) >= 3 and (
        candidate in target or target in candidate
    )


def mentions_concrete_detail(text: str) -> bool:
    """
    Heuristic: did the candidate mention a project, system, or specific experience?

    Used to decide whether a strong answer earns one curiosity follow-up before
    difficulty ramps. No LLM call.
    """
    normalized = " ".join(str(text or "").split())
    if len(normalized.split()) < 8:
        return False
    verb_hit = bool(_PROJECT_VERBS.search(normalized))
    marker_hit = bool(_CONCRETE_DETAIL_MARKERS.search(normalized))
    return verb_hit and marker_hit


def opening_resume_skills(state: InterviewState, *, limit: int = 2) -> list[str]:
    """
    Return up to ``limit`` skills present on both the resume and the interview plan.

    Order follows plan section order for stable, predictable openings.
    """
    resume_skills = [
        str(item).strip()
        for item in (state.get("resume_context") or {}).get("skills", [])
        if str(item).strip()
    ]
    if not resume_skills:
        return []

    plan_skills: list[str] = []
    seen: set[str] = set()
    for section in state.get("runtime_sections") or []:
        if section.get("section_kind") != "technical":
            continue
        skill = str(section.get("skill") or section.get("section_name") or "").strip()
        key = skill.casefold()
        if skill and key not in seen:
            plan_skills.append(skill)
            seen.add(key)

    matched: list[str] = []
    for plan_skill in plan_skills:
        if any(
            _skills_match(resume_skill, plan_skill) for resume_skill in resume_skills
        ):
            matched.append(plan_skill)
        if len(matched) >= limit:
            break
    return matched


def format_resume_skills_phrase(skills: list[str]) -> str:
    """Format one or two skills for natural spoken insertion."""
    if not skills:
        return ""
    if len(skills) == 1:
        return skills[0]
    return f"{skills[0]} and {skills[1]}"


def resume_has_skill(state: InterviewState, skill: str | None) -> bool:
    """
    Use tolerant case-insensitive matching for normalized resume skills.

    Args:
        state: The current interview state.
        skill: The specific technical skill to look for.

    Returns:
        True if a matching skill is found on the candidate's resume.
    """

    target = str(skill or "").strip()
    if not target:
        return False
    for raw in (state.get("resume_context") or {}).get("skills", []):
        if _skills_match(str(raw), target):
            return True
    return False


def target_section(state: InterviewState) -> tuple[int, dict[str, Any]]:
    """
    Return the next section selected by normal progression or forced timing control.

    Args:
        state: The current interview state.

    Returns:
        A tuple of (section_index, section_dictionary).

    Raises:
        RuntimeError: If no sections are available in the state.
    """

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
    """
    Move one difficulty level up (+1) or down (-1) within the bounded range (easy, medium, hard).

    Args:
        current: The current difficulty level.
        direction: Integer step (-1 for easier, +1 for harder).

    Returns:
        The new difficulty level.
    """
    index = DIFFICULTIES.index(current)
    return DIFFICULTIES[min(2, max(0, index + direction))]


def determine_question_difficulty(
    state: InterviewState,
    *,
    skill: str,
    entering_new_section: bool,
) -> tuple[Difficulty, bool]:
    """
    Determine target difficulty and whether this is the single weak-answer probe.
    Adjusts question difficulty adaptively based on the candidate's streak of
    recent live evaluation results for this specific skill.

    Args:
        state: The current interview state.
        skill: The technical skill currently being tested.
        entering_new_section: Whether the interview just transitioned to a new section.

    Returns:
        A tuple of (Difficulty, probe_deeper_flag).
    """

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


def difficulty_plan(
    state: InterviewState,
    *,
    skill: str,
    entering_new_section: bool,
) -> dict[str, dict[str, Any]]:
    """
    Pre-compute the deterministic difficulty/probe outcome for every possible
    strength judgement of the current answer.

    The merged interviewer call judges the answer strength itself; the server then
    enforces the difficulty mapped here so adaptive difficulty stays fully
    deterministic and never depends on model-chosen numbers.

    Args:
        state: The current interview state.
        skill: The technical skill currently being tested.
        entering_new_section: Whether the next question opens a new section.

    Returns:
        A mapping of "weak"/"adequate"/"strong" to {"difficulty", "probe_deeper"}.
    """

    resume_floor = resume_has_skill(state, skill)
    initial = ROLE_INITIAL_DIFFICULTY.get(
        str(state.get("inferred_difficulty") or "").lower(),
        "medium",
    )

    def _floor(value: Difficulty) -> Difficulty:
        return "medium" if (resume_floor and value == "easy") else value

    if entering_new_section or state.get("current_question_difficulty") is None:
        base = {
            "difficulty": _floor(initial),
            "probe_deeper": False,
            "follow_interesting_thread": False,
        }
        return {strength: dict(base) for strength in ("weak", "adequate", "strong")}

    current: Difficulty = state.get("current_question_difficulty") or initial
    streaks = (state.get("skill_evaluation_streaks") or {}).get(skill.casefold(), {})
    prior_weak = int(streaks.get("weak") or 0)
    prior_adequate = int(streaks.get("adequate") or 0)
    prior_response = str(state.get("previous_candidate_response") or "")
    thread_eligible = not state.get(
        "thread_follow_up_used"
    ) and mentions_concrete_detail(prior_response)

    return {
        "weak": {
            "difficulty": _floor(_one_step(current, -1)),
            # The first weak answer in a row earns the one permitted focused probe.
            "probe_deeper": (prior_weak + 1) == 1,
            "follow_interesting_thread": False,
        },
        "adequate": {
            "difficulty": _floor(
                _one_step(current, 1) if (prior_adequate + 1) >= 2 else current
            ),
            "probe_deeper": False,
            "follow_interesting_thread": False,
        },
        "strong": {
            "difficulty": current if thread_eligible else _floor(_one_step(current, 1)),
            "probe_deeper": False,
            "follow_interesting_thread": thread_eligible,
        },
    }
