"""Load and normalize assessment context for interview sessions."""

from __future__ import annotations

from typing import Any


def normalize_sections(
    interview_plan: dict | None, duration_mins: int | None
) -> list[dict]:
    plan = interview_plan or {}
    raw_sections = plan.get("sections") if isinstance(plan, dict) else None
    if not isinstance(raw_sections, list) or not raw_sections:
        total = float(duration_mins or plan.get("total_mins") or 5)
        if int(total) == 5:
            raw_sections = [
                {
                    "section_name": "self_intro",
                    "skill": None,
                    "allocated_mins": 1.0,
                    "priority_score": None,
                },
                {
                    "section_name": "behavioural",
                    "skill": None,
                    "allocated_mins": 1.0,
                    "priority_score": None,
                },
                {
                    "section_name": "cultural",
                    "skill": None,
                    "allocated_mins": 1.0,
                    "priority_score": None,
                },
                {
                    "section_name": "technical",
                    "skill": "Technical",
                    "allocated_mins": 2.0,
                    "priority_score": 10.0,
                },
            ]
        else:
            raw_sections = [
                {
                    "section_name": "self_intro",
                    "skill": None,
                    "allocated_mins": max(1.0, round(total * 0.2, 1)),
                    "priority_score": None,
                },
                {
                    "section_name": "behavioural",
                    "skill": None,
                    "allocated_mins": max(1.0, round(total * 0.4, 1)),
                    "priority_score": None,
                },
                {
                    "section_name": "cultural",
                    "skill": None,
                    "allocated_mins": max(1.0, round(total * 0.4, 1)),
                    "priority_score": None,
                },
            ]

    normalized: list[dict] = []
    for idx, raw in enumerate(raw_sections):
        section = raw if isinstance(raw, dict) else {}
        name = (
            section.get("section_name")
            or section.get("name")
            or section.get("skill")
            or f"section_{idx + 1}"
        )
        allocated_mins = section.get("allocated_mins")
        if allocated_mins is None and section.get("time_budget_secs") is not None:
            allocated_mins = float(section["time_budget_secs"]) / 60.0
        allocated_mins = float(allocated_mins or 1.0)
        skill = section.get("skill")
        if skill in {"", "none", "None"}:
            skill = None
        normalized.append(
            {
                **section,
                "section_name": str(name),
                "name": str(name),
                "skill": skill,
                "allocated_mins": allocated_mins,
                "time_budget_secs": int(allocated_mins * 60),
                "priority_score": section.get("priority_score"),
            }
        )
    return normalized


def extract_resume_skills(resume: dict | None) -> list[str]:
    data: dict[str, Any] = resume or {}
    skills = data.get("skills") or data.get("technical_skills") or []
    if isinstance(skills, dict):
        flattened: list[str] = []
        for value in skills.values():
            if isinstance(value, list):
                flattened.extend(str(item) for item in value)
            elif value:
                flattened.append(str(value))
        return flattened
    if isinstance(skills, list):
        return [str(item) for item in skills if item]
    return []


def normalize_resume(resume: dict | None) -> dict:
    data = dict(resume or {})
    data["skills"] = extract_resume_skills(data)
    data.setdefault(
        "experience_years",
        data.get("years_of_experience")
        or data.get("total_experience_years")
        or "unknown",
    )
    return data
