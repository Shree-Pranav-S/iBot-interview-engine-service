"""Tests for self-introduction turn handling."""

from __future__ import annotations

import re

from src.control.agents.nodes.time_manager import decide_time_action

REPEAT_PATTERN = re.compile(
    r"\b("
    r"repeat (?:it|that|this question|that question|the question)|"
    r"can you repeat (?:it|that|the question|this question)|"
    r"could you repeat (?:it|that|the question|this question)|"
    r"say (?:it|that) again|hear (?:it|that|this question|the question) again|"
    r"once more"
    r")\b",
    re.IGNORECASE,
)


def _is_self_intro_phase(state: dict) -> bool:
    return bool(state.get("is_self_introduction")) or (
        state.get("current_section_kind") == "self_intro"
    )


def _intro_state(**overrides):
    base = {
        "current_section_kind": "self_intro",
        "is_self_introduction": True,
        "current_section": "self_intro",
        "current_section_index": 0,
        "current_section_budget_secs": 120,
        "current_section_started_elapsed_secs": 0,
        "elapsed_secs": 10,
        "total_duration_secs": 1800,
        "section_budgets_secs": {"0": 120, "1": 600},
        "runtime_sections": [
            {
                "section_kind": "self_intro",
                "section_name": "self_intro",
                "allocated_mins": 2,
            },
            {
                "section_kind": "technical",
                "section_name": "python",
                "skill": "Python",
                "allocated_mins": 10,
            },
        ],
    }
    base.update(overrides)
    return base


class TestDecideTimeActionSelfIntro:
    def test_substantial_intro_still_transitions_on_completed_turn(self) -> None:
        decision = decide_time_action(_intro_state())
        assert decision["action"] == "transition"
        assert decision["transition_reason"] == "self_introduction_complete"


class TestRepeatPattern:
    def test_can_you_repeat_the_question_matches(self) -> None:
        assert REPEAT_PATTERN.search("Can you repeat the question?")


class TestSelfIntroPhaseHelper:
    def test_detects_self_intro(self) -> None:
        assert _is_self_intro_phase({"is_self_introduction": True})
        assert _is_self_intro_phase({"current_section_kind": "self_intro"})
        assert not _is_self_intro_phase({"current_section_kind": "technical"})
