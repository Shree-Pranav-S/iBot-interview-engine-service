"""Node for updating skill tracking state."""

from __future__ import annotations

from statistics import mean
from typing import Any

from src.control.agents.state import InterviewState


def update_skill_state(state: InterviewState) -> dict[str, Any]:
    evaluation = state.get("latest_evaluation") or {}
    section = str(
        evaluation.get("section") or state.get("current_section") or "general"
    )
    skill = str(evaluation.get("skill") or section)
    key = skill.lower()
    progress = dict(state.get("skill_progress") or {})
    current = dict(progress.get(key) or {})
    scores = [
        *list(current.get("scores") or []),
        float(evaluation.get("provisional_score") or 0),
    ]
    observed = sorted(
        set(current.get("signals_observed") or [])
        | set(evaluation.get("signals_observed") or [])
    )
    missing = sorted(set(evaluation.get("signals_missing") or []))
    strength = str(evaluation.get("strength") or "").lower()
    is_strong = (
        strength == "strong" or float(evaluation.get("provisional_score") or 0) >= 4.0
    )
    is_adequate = strength == "adequate"
    consecutive_strong = int(current.get("consecutive_strong_answers") or 0)
    consecutive_strong = consecutive_strong + 1 if is_strong else 0
    consecutive_adequate = int(current.get("consecutive_adequate_answers") or 0)
    consecutive_adequate = consecutive_adequate + 1 if is_adequate else 0
    progress[key] = {
        "skill": skill,
        "section": section,
        "answers": int(current.get("answers") or 0) + 1,
        "scores": scores[-5:],
        "average_score": round(mean(scores), 2),
        "best_score": max(scores),
        "signals_observed": observed,
        "signals_missing": missing,
        "consecutive_strong_answers": consecutive_strong,
        "consecutive_adequate_answers": consecutive_adequate,
        "last_strength": strength,
    }
    return {"skill_progress": progress, "next_node": None}
