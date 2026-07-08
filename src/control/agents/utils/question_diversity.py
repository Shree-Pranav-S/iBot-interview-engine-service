"""Deterministic framing hints and server-side question quality validators."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from src.control.agents.state import InterviewState
from src.core.exceptions import QuestionValidationException

_FRAMING_MODES = (
    "concrete_scenario",
    "mechanism_explanation",
    "comparison",
    "failure_mode",
    "design_tradeoff",
    "debugging_steps",
    "edge_case",
    "operational_impact",
)

_QUESTION_STOP_WORDS = {
    "a",
    "an",
    "and",
    "can",
    "could",
    "describe",
    "do",
    "explain",
    "how",
    "in",
    "is",
    "of",
    "the",
    "to",
    "what",
    "when",
    "with",
    "would",
    "you",
    "your",
}

_PLACEHOLDER_PATTERNS = (
    r"practical\s+.+\s+use",
    r"a\s+problem\s+related\s+to",
    r"an\s+approach\s+involving",
    r"a\s+real[- ]world\s+.+\s+issue",
    r"applying\s+(\w+)\s+in\s+\1",
    r"(\w+)\s+use\s+(with|in)\s+\1",
)


def framing_hint(seed: str, sequence_number: int) -> str:
    """Rotate a deterministic framing mode from seed and turn sequence."""

    digest = hashlib.sha256(f"{seed}:{sequence_number}".encode()).hexdigest()
    index = int(digest[:8], 16) % len(_FRAMING_MODES)
    return _FRAMING_MODES[index]


def recent_question_stems(state: InterviewState, *, limit: int = 3) -> list[str]:
    """Return opening stems (first four words) of recent asked questions."""

    stems: list[str] = []
    for item in state.get("asked_questions") or []:
        question = str(item.get("question_text") or "").strip()
        if not question:
            continue
        words = question.split()[:4]
        if words:
            stems.append(" ".join(words).casefold())
    return stems[-limit:]


def recent_acknowledgements(
    state: InterviewState,
    *,
    limit: int,
) -> list[str]:
    """Return recent non-empty interviewer acknowledgement phrases."""

    acknowledgements = [
        str(item.get("acknowledgement") or "").strip()
        for item in state.get("asked_questions") or []
        if str(item.get("acknowledgement") or "").strip()
    ]
    return acknowledgements[-limit:]


def normalize_question_text(value: str) -> str:
    """Normalize a question for similarity and duplicate comparisons."""

    return " ".join(re.findall(r"[a-z0-9+#.]+", value.casefold()))


def _question_tokens(value: str) -> set[str]:
    return {
        token
        for token in normalize_question_text(value).split()
        if token not in _QUESTION_STOP_WORDS
    }


def _question_stem(value: str) -> str:
    words = value.strip().split()[:4]
    return " ".join(words).casefold() if words else ""


def validate_unique_question(
    question_text: str,
    asked_questions: list[dict[str, Any]],
    *,
    allow_related_probe: bool = False,
) -> None:
    """Reject exact repeats and high token overlap with prior questions."""

    normalized = normalize_question_text(question_text)
    new_tokens = _question_tokens(question_text)
    for item in asked_questions:
        prior_text = str(item.get("question_text") or "")
        if normalized == normalize_question_text(prior_text):
            raise QuestionValidationException(
                "question generator repeated a previous question"
            )
        if allow_related_probe:
            continue
        prior_tokens = _question_tokens(prior_text)
        union = new_tokens | prior_tokens
        if union and len(new_tokens & prior_tokens) / len(union) >= 0.80:
            raise QuestionValidationException(
                "question generator lightly paraphrased a previous question"
            )


def validate_topic_fresh(
    topic: str,
    used_topics: list[str],
    *,
    allow_related_probe: bool = False,
) -> None:
    """Reject when the topic label was already used for this skill."""

    if allow_related_probe:
        return
    normalized = topic.strip().casefold()
    if normalized and normalized in {t.casefold() for t in used_topics}:
        raise QuestionValidationException("question generator reused a prior topic")


def validate_no_placeholder_phrasing(question_text: str, skill: str) -> None:
    """Reject vague placeholder or tautological skill wording."""

    lowered = question_text.casefold()
    skill_fold = skill.casefold()
    for pattern in _PLACEHOLDER_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            raise QuestionValidationException(
                "question generator used placeholder phrasing"
            )
    if skill_fold and (
        f"practical {skill_fold} use" in lowered
        or f"{skill_fold} use in {skill_fold}" in lowered
        or f"applying {skill_fold} in {skill_fold}" in lowered
    ):
        raise QuestionValidationException(
            "question generator repeated the skill tautologically"
        )


def validate_stem_diversity(
    question_text: str,
    asked_questions: list[dict[str, Any]],
    *,
    lookback: int = 2,
) -> None:
    """Reject when the opening stem matches one of the last N questions."""

    stem = _question_stem(question_text)
    if not stem:
        return
    recent = [
        _question_stem(str(item.get("question_text") or ""))
        for item in (asked_questions or [])[-lookback:]
    ]
    if stem in {s for s in recent if s}:
        raise QuestionValidationException(
            "question generator repeated a recent opening stem"
        )


def validate_generated_question(
    *,
    question_text: str,
    topic: str,
    skill: str,
    asked_questions: list[dict[str, Any]],
    used_topics: list[str],
    allow_related_probe: bool = False,
) -> None:
    """Run all question-quality validators."""

    validate_unique_question(
        question_text,
        asked_questions,
        allow_related_probe=allow_related_probe,
    )
    validate_topic_fresh(
        topic,
        used_topics,
        allow_related_probe=allow_related_probe,
    )
    validate_no_placeholder_phrasing(question_text, skill)
    validate_stem_diversity(question_text, asked_questions)
