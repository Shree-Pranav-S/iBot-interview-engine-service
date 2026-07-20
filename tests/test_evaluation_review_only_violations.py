"""Evaluation isolation for uncalibrated review-only proctoring signals."""

from src.core.services.evaluation_context_builder import (
    _violation_affects_evaluation,
)


def test_authoritative_proctoring_categories_override_legacy_review_only_flag() -> None:
    assert _violation_affects_evaluation({"violation_type": "tab_switch"}) is True
    assert (
        _violation_affects_evaluation(
            {
                "violation_type": "prompt_injection",
                "metadata": {"affects_evaluation": True},
            }
        )
        is True
    )
    assert (
        _violation_affects_evaluation(
            {
                "violation_type": "face_absent",
                "metadata": {"affects_evaluation": False},
            }
        )
        is True
    )


def test_unrelated_explicit_review_only_violation_is_still_excluded() -> None:
    assert (
        _violation_affects_evaluation(
            {
                "violation_type": "experimental_signal",
                "metadata": {"affects_evaluation": False},
            }
        )
        is False
    )
