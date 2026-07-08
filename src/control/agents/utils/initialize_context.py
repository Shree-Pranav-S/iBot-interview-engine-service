"""Helpers for loading interview context before graph speech begins."""

from __future__ import annotations

from typing import Any

from src.core.exceptions import InterviewPlanInvalidException


def _skills(value: Any) -> list[str]:
    """
    Extract a unique list of candidate skills from the resume payload.

    Args:
        value: A list of string skills or dict objects containing skill names.

    Returns:
        A normalized, deduplicated list of strings.
    """
    if not isinstance(value, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, str):
            skill = item.strip()
        elif isinstance(item, dict):
            skill = str(
                item.get("skill") or item.get("name") or item.get("title") or ""
            ).strip()
        else:
            skill = ""

        key = skill.casefold()
        if skill and key not in seen:
            normalized.append(skill)
            seen.add(key)
    return normalized


def _experience_years(value: Any) -> float:
    """
    Safely extract candidate experience years from the resume payload.

    Args:
        value: A raw number or string representing years of experience.

    Returns:
        A float representing years, defaulting to 0.0 on error.
    """
    try:
        return max(0.0, float(value or 0))
    except (TypeError, ValueError):
        return 0.0


def _runtime_sections(interview_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Normalize the stored interview plan into an actionable sequence of runtime sections.
    This ensures each section has an explicitly defined kind (technical, behavioural)
    and validates time allocations.

    Args:
        interview_plan: The stored plan dictionary for the candidate.

    Returns:
        A standardized list of section dictionaries.

    Raises:
        InterviewPlanInvalidException: If the plan is malformed or missing sections.
    """
    raw_sections = interview_plan.get("sections")
    if not isinstance(raw_sections, list) or not raw_sections:
        raise InterviewPlanInvalidException(
            "Interview plan must contain at least one section"
        )

    sections: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_sections):
        if not isinstance(raw, dict):
            continue
        section_name = str(raw.get("section_name") or "").strip()
        skill = str(raw.get("skill") or "").strip() or None
        if section_name == "self_intro":
            kind = "self_intro"
        elif section_name == "behavioural_cultural":
            kind = "behavioural_cultural"
        else:
            kind = "technical"
            if not skill:
                skill = section_name

        expected = raw.get("expected_signals")
        expected_signals = [
            str(item).strip()
            for item in (expected if isinstance(expected, list) else [])
            if str(item).strip()
        ]
        sections.append(
            {
                "index": index,
                "section_name": section_name or f"section_{index + 1}",
                "section_kind": kind,
                "skill": skill,
                "expected_signals": expected_signals,
                "allocated_mins": float(raw.get("allocated_mins") or 0),
            }
        )

    if not sections:
        raise InterviewPlanInvalidException(
            "Interview plan sections could not be normalized"
        )
    return sections
