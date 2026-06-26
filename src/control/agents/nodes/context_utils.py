"""Context extraction helpers for interview graph nodes."""

from __future__ import annotations

import re
from typing import Any

from src.control.agents.state import InterviewState

BEHAVIOURAL_CULTURAL_SECTION_KEYS = {
    "behavioural_cultural",
    "behavioral_cultural",
    "behavioural",
    "behavioral",
    "cultural",
    "culture",
}
NON_TECH_SECTIONS = {
    "self_intro",
    *BEHAVIOURAL_CULTURAL_SECTION_KEYS,
}


def _next_bot_turn_number(state: InterviewState) -> int:
    pending_candidate = state.get("pending_candidate_turn")
    if pending_candidate:
        return int(pending_candidate.get("turn_number") or 0) + 1
    return int(state.get("turn_number") or 0) + 1


NO_EXPERIENCE_PATTERNS = (
    "no experience",
    "haven't used",
    "have not used",
    "never used",
    "don't know",
    "do not know",
    "not familiar",
    "not worked",
    "not worked with",
)
RESUME_SKILL_KEYS = ("skills", "technical_skills", "technologies", "tools")
TECHNICAL_SUBSTANCE_MARKERS = {
    "api",
    "async",
    "auth",
    "cache",
    "database",
    "db",
    "docker",
    "endpoint",
    "http",
    "index",
    "latency",
    "migration",
    "query",
    "queue",
    "schema",
    "service",
    "sql",
    "transaction",
}
MEANINGFUL_ATTEMPT_MARKERS = (
    " but ",
    " however ",
    " i would ",
    " i'd ",
    " i will ",
    " my approach ",
    " from my understanding ",
    " i think ",
    " we can ",
    " we could ",
    " for example ",
    " in practice ",
)


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_#+.-]+", text or ""))


def section_key(value: Any) -> str:
    """Normalize a section label for comparisons across stored plan variants."""

    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def is_self_intro_section(state: InterviewState) -> bool:
    return section_key(state.get("current_section")) == "self_intro"


def is_behavioural_cultural_section_name(value: Any) -> bool:
    return section_key(value) in BEHAVIOURAL_CULTURAL_SECTION_KEYS


def is_behavioural_cultural_section(state: InterviewState) -> bool:
    return is_behavioural_cultural_section_name(state.get("current_section"))


def behavioural_cultural_section_index(state: InterviewState) -> int | None:
    for index, section in enumerate(state.get("section_order") or []):
        if is_behavioural_cultural_section_name(section):
            return index
    return None


def is_technical_section(state: InterviewState) -> bool:
    section = section_key(state.get("current_section"))
    if section in NON_TECH_SECTIONS:
        return False
    return bool(state.get("current_skill")) or section not in {"", "general"}


def behavioural_cultural_requirement_met(state: InterviewState) -> bool:
    """Return true once the combined non-technical section has Q and response evidence."""

    asked = any(
        is_behavioural_cultural_section_name(item.get("section"))
        for item in state.get("asked_questions") or []
        if isinstance(item, dict)
    )
    if not asked:
        return False

    evaluated = any(
        is_behavioural_cultural_section_name(item.get("section"))
        for item in state.get("live_evaluations") or []
        if isinstance(item, dict)
    )
    if evaluated:
        return True

    pending = state.get("pending_candidate_turn")
    if isinstance(pending, dict) and is_behavioural_cultural_section_name(
        pending.get("section")
    ):
        response_type = str((pending.get("metadata") or {}).get("response_type") or "")
        return response_type not in {"", "timer_expired", "silence"}

    return False


def resume_mentions_skill(resume_parsed: dict[str, Any], skill: str | None) -> bool:
    if not skill:
        return False
    lowered = skill.lower()
    haystacks: list[str] = []
    if isinstance(resume_parsed, dict):
        for key in ("skills", "technical_skills", "technologies", "tools"):
            value = resume_parsed.get(key)
            if isinstance(value, list):
                haystacks.extend(str(item) for item in value)
            elif value:
                haystacks.append(str(value))
        haystacks.append(str(resume_parsed))
    return any(lowered in item.lower() for item in haystacks)


def extract_resume_skills(resume_parsed: dict[str, Any]) -> list[str]:
    """Return skill-like values found in the parsed resume."""

    if not isinstance(resume_parsed, dict):
        return []

    skills: list[str] = []
    for key in RESUME_SKILL_KEYS:
        value = resume_parsed.get(key)
        if isinstance(value, list):
            skills.extend(str(item).strip() for item in value if str(item).strip())
        elif value:
            skills.extend(
                item.strip() for item in re.split(r"[,;/|]", str(value)) if item.strip()
            )

    seen: set[str] = set()
    unique: list[str] = []
    for skill in skills:
        lowered = skill.lower()
        if lowered not in seen:
            seen.add(lowered)
            unique.append(skill)
    return unique


def answer_claims_no_experience(text: str, skill: str | None = None) -> bool:
    lowered = (text or "").lower()
    if not any(pattern in lowered for pattern in NO_EXPERIENCE_PATTERNS):
        return False
    return not skill or skill.lower() in lowered or "it" in lowered


def is_substantial_answer(state: InterviewState, text: str) -> tuple[bool, str]:
    words = word_count(text)
    lowered = (text or "").strip().lower()
    padded_lowered = f" {lowered} "
    filler_answers = {
        "yes",
        "no",
        "maybe",
        "ok",
        "okay",
        "sure",
        "fine",
        "good",
        "bad",
        "nothing",
        "none",
        "na",
        "n/a",
    }
    if not lowered or lowered in filler_answers:
        return False, "answer_too_short"
    claims_no_experience = answer_claims_no_experience(text, state.get("current_skill"))
    has_meaningful_attempt = words >= 12 and any(
        marker in padded_lowered for marker in MEANINGFUL_ATTEMPT_MARKERS
    )
    if claims_no_experience and not has_meaningful_attempt:
        return False, "candidate_claimed_no_experience"

    if is_self_intro_section(state) or is_behavioural_cultural_section(state):
        return (words > 10, "substantial" if words > 10 else "answer_under_10_words")

    if is_technical_section(state):
        if words <= 1:
            return False, "technical_answer_too_short"
        if words <= 3 and not any(
            marker in lowered for marker in TECHNICAL_SUBSTANCE_MARKERS
        ):
            return False, "technical_answer_lacks_substance"
        return True, "substantial"

    return (words > 10, "substantial" if words > 10 else "answer_under_10_words")
