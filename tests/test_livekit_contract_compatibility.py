"""Compatibility checks for cross-service LiveKit authorization payloads."""

from datetime import UTC, datetime
from uuid import uuid4

from src.schemas.livekit import CandidateConnectionContext


def test_candidate_connection_context_accepts_additive_core_api_fields() -> None:
    context = CandidateConnectionContext.model_validate(
        {
            "session_id": uuid4(),
            "connection_id": "connection-1",
            "candidate_assessment_id": uuid4(),
            "candidate_id": uuid4(),
            "assessment_id": uuid4(),
            "candidate_name": "Candidate",
            "session_token_expires_at": datetime.now(UTC),
            "elapsed_secs": 12,
            "interview_started": True,
            "tab_switch_count": 2,
            "future_optional_context": "safe-to-ignore",
        }
    )

    assert context.tab_switch_count == 2
    assert context.candidate_name == "Candidate"
