"""Evaluation readiness for interviews terminated by proctoring policy."""

from src.core.services.evaluation_service import READY_SESSION_STATUSES


def test_terminated_interviews_are_ready_for_holistic_evaluation() -> None:
    assert "TERMINATED" in READY_SESSION_STATUSES
