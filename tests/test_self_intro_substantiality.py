"""Tests for cumulative self-introduction substantiality checks."""

from __future__ import annotations

from src.control.agents.nodes.classify_response import (
    SELF_INTRO_ELABORATION_WORD_THRESHOLD,
    _deterministic_classification,
    self_intro_is_substantial,
)


def _intro_state(**overrides):
    base = {
        "current_section_kind": "self_intro",
        "self_intro_accumulated_response": "",
    }
    base.update(overrides)
    return base


def test_split_intro_turns_become_substantial_when_combined() -> None:
    first = "Yeah sure my name is Alex and I work on backend systems."
    second = "I have built APIs with Python and PostgreSQL for two years."

    state = _intro_state()
    assert not self_intro_is_substantial(state, first)

    state["self_intro_accumulated_response"] = first
    assert self_intro_is_substantial(state, second)


def test_silence_after_substantial_intro_is_treated_as_answer() -> None:
    prior = " ".join(["word"] * SELF_INTRO_ELABORATION_WORD_THRESHOLD)
    state = _intro_state(self_intro_accumulated_response=prior)

    result = _deterministic_classification(
        {**state, "candidate_event": {"event_type": "silence_timeout"}},  # type: ignore[typeddict-item]
        "",
    )

    assert result is not None
    assert result["response_type"] == "answer"
    assert result["is_substantial"] is True
