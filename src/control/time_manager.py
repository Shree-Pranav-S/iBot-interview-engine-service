"""Timer helpers for the interview graph."""

from __future__ import annotations

import time

from src.control.agents.state import InterviewState


def compute_elapsed(state: InterviewState) -> int:
    started_at = float(state.get("interview_started_at") or time.time())
    total_pause_secs = int(state.get("total_pause_secs") or 0)
    return max(0, int(time.time() - started_at - total_pause_secs))


def total_allocated_secs(state: InterviewState) -> int:
    explicit = int(state.get("total_interview_allocated_secs") or 0)
    if explicit > 0:
        return explicit
    return sum(
        int(section.get("time_budget_secs") or 0)
        for section in state.get("sections") or []
    )


def compute_section_elapsed(state: InterviewState) -> int:
    started_at = float(state.get("section_started_at") or time.time())
    return max(0, int(time.time() - started_at))


def compute_section_time_remaining(state: InterviewState) -> float:
    allocated = float(state.get("section_allocated_secs") or 60.0)
    return max(0.0, allocated - compute_section_elapsed(state))


def max_questions_for_section(section: dict) -> int:
    name = str(section.get("section_name") or section.get("name") or "").lower()
    allocated_mins = float(section.get("allocated_mins") or 1.0)
    if name == "self_intro":
        return 1
    if name in {"behavioural", "behavioral", "cultural"}:
        return max(1, min(2, int(round(allocated_mins))))
    return max(1, min(4, int(allocated_mins // 1.5) + 1))


def check_section_should_advance(state: InterviewState) -> bool:
    sections = state.get("sections") or []
    idx = int(state.get("current_section_index") or 0)
    if idx >= len(sections):
        return True

    questions_asked = int(state.get("questions_asked_in_section") or 0)
    if questions_asked <= 0:
        return False

    if compute_section_time_remaining(state) <= 0:
        return True

    return questions_asked >= max_questions_for_section(sections[idx])  # type: ignore


def check_interview_complete(state: InterviewState) -> bool:
    sections = state.get("sections") or []
    if not sections:
        return True
    if bool(state.get("auto_submit_triggered")):
        return True
    total_secs = total_allocated_secs(state)
    if total_secs > 0 and compute_elapsed(state) >= total_secs:
        return True
    idx = int(state.get("current_section_index") or 0)
    return idx >= len(sections)


def next_difficulty_from_quality(
    current: int, quality: str, priority: float | None = None
) -> int:
    level = int(current or 2)
    if quality == "strong":
        level += 1
    elif quality in {"weak", "non_answer"}:
        level -= 1
    if priority is not None and float(priority or 0) <= 4:
        level = min(level, 2)
    return max(1, min(3, level))


def difficulty_label(level: int) -> str:
    return {1: "easy", 2: "medium", 3: "hard"}.get(int(level or 2), "medium")


def get_time_pressure_hint(remaining_secs: float) -> str:
    if remaining_secs < 60:
        return "Less than 1 minute remains; ask a quick wrap-up question."
    if remaining_secs < 120:
        return "Under 2 minutes remain; keep the question concise."
    return ""
