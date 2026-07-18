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


def _bounded_text(value: object, *, max_chars: int) -> str:
    """Normalize one compact prompt string to a hard character bound."""

    return " ".join(str(value or "").split())[:max_chars].strip()


def _bounded_list(
    value: object,
    *,
    max_items: int = 6,
    max_chars: int = 120,
) -> list[str]:
    """Normalize a bounded, deduplicated list for live prompt context."""

    if not isinstance(value, list):
        return []
    items: list[str] = []
    seen: set[str] = set()
    for raw in value:
        item = _bounded_text(raw, max_chars=max_chars)
        key = item.casefold()
        if item and key not in seen:
            items.append(item)
            seen.add(key)
        if len(items) >= max_items:
            break
    return items


def _analysis_skills(jd_analysis: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return normalized JD skill records used only for legacy-plan fallbacks."""

    if not isinstance(jd_analysis, dict):
        return []
    skills = jd_analysis.get("skills")
    if not isinstance(skills, list):
        return []
    return [item for item in skills if isinstance(item, dict)]


def _runtime_sections(
    interview_plan: dict[str, Any],
    *,
    role_name: str = "",
    jd_analysis: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
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
    analysis_skills = _analysis_skills(jd_analysis)
    planned_skills = {
        str(raw.get("skill") or raw.get("section_name") or "").strip().casefold()
        for raw in raw_sections
        if isinstance(raw, dict)
        and str(raw.get("section_name") or "")
        not in {"self_intro", "behavioural_cultural"}
    }
    omitted_skills = [
        _bounded_text(item.get("skill"), max_chars=80)
        for item in analysis_skills
        if _bounded_text(item.get("skill"), max_chars=80).casefold()
        not in planned_skills
    ][:6]
    role_level = _bounded_text(
        interview_plan.get("inferred_difficulty") or "mid-level",
        max_chars=80,
    )
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
        question_brief: dict[str, Any] | None = None
        if kind == "technical" and skill:
            raw_brief = raw.get("question_brief")
            brief = raw_brief if isinstance(raw_brief, dict) else {}
            matching_analysis = next(
                (
                    item
                    for item in analysis_skills
                    if str(item.get("skill") or "").strip().casefold()
                    == skill.casefold()
                ),
                {},
            )
            responsibility = _bounded_text(
                brief.get("role_responsibility")
                or matching_analysis.get("reasoning")
                or f"Apply {skill} in the responsibilities of the {role_name or 'target'} role.",
                max_chars=320,
            )
            environment = _bounded_text(
                brief.get("operating_environment")
                or f"{role_name or 'Target'} role; no narrower environment was stored.",
                max_chars=240,
            )
            important_tools = _bounded_list(
                brief.get("important_tools") if "important_tools" in brief else [skill]
            )
            out_of_scope_source = (
                brief.get("out_of_scope_topics")
                if "out_of_scope_topics" in brief
                else omitted_skills
            )
            question_brief = {
                "expected_signals": expected_signals[:4],
                "role_responsibility": responsibility,
                "operating_environment": environment,
                "important_tools": important_tools,
                "constraints": _bounded_list(brief.get("constraints")),
                "seniority_depth": _bounded_text(
                    brief.get("seniority_depth")
                    or f"Use {role_level} reasoning depth for this responsibility.",
                    max_chars=240,
                ),
                "out_of_scope_topics": _bounded_list(out_of_scope_source),
            }
        sections.append(
            {
                "index": index,
                "section_name": section_name or f"section_{index + 1}",
                "section_kind": kind,
                "skill": skill,
                "expected_signals": expected_signals,
                "question_brief": question_brief,
                "allocated_mins": float(raw.get("allocated_mins") or 0),
            }
        )

    if not sections:
        raise InterviewPlanInvalidException(
            "Interview plan sections could not be normalized"
        )
    return sections
