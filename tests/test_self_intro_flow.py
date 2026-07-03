"""Tests for self-introduction turn handling."""

from __future__ import annotations

from src.control.agents.nodes.classify_response import REPEAT_PATTERN
from src.control.agents.nodes.question_strategy import is_self_intro_phase
from src.control.agents.nodes.time_manager import decide_time_action


def _intro_state(**overrides):
    base = {
        "current_section_kind": "self_intro",
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
        assert is_self_intro_phase({"current_section_kind": "self_intro"})
        assert not is_self_intro_phase({"current_section_kind": "technical"})
