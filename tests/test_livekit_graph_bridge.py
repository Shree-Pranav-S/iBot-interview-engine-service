"""Behavior contracts for the LiveKit-to-LangGraph bridge."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import src.core.services.livekit_graph_bridge as bridge_module
from src.core.services.livekit_graph_bridge import LiveKitInterviewBridge


@pytest.mark.parametrize(
    ("submission", "expected_event"),
    [
        (
            "candidate",
            {
                "event_type": "candidate_answer",
                "text": "hello world",
                "stt_confidence": 0.9,
                "duration_ms": 1250,
            },
        ),
        (
            "silence",
            {
                "event_type": "silence_timeout",
                "text": "",
                "silence_duration_ms": 5000,
            },
        ),
        (
            "barge_in",
            {
                "event_type": "section_time_barge_in",
                "text": "partial answer",
            },
        ),
    ],
)
async def test_graph_submissions_share_the_same_resume_contract(
    monkeypatch: pytest.MonkeyPatch,
    submission: str,
    expected_event: dict[str, object],
) -> None:
    graph = SimpleNamespace(
        ainvoke=AsyncMock(
            return_value={
                "bot_reply_text": "  next reply  ",
                "timer_started": False,
                "timer_started_at": None,
            }
        )
    )
    get_graph = AsyncMock(return_value=graph)
    monkeypatch.setattr(bridge_module, "get_graph", get_graph)

    bridge = LiveKitInterviewBridge(
        candidate_assessment_id="candidate-assessment-id",
        interview_session_id="interview-session-id",
        connection_id="connection-id",
    )
    bridge.state = {
        "session_status": "IN_PROGRESS",
        "elapsed_secs": 12,
        "timer_started": True,
        "timer_started_at": "2026-01-01T00:00:00+00:00",
    }

    if submission == "candidate":
        reply = await bridge.submit_candidate_turn(
            "  hello   world  ",
            stt_confidence=0.9,
            duration_ms=1250,
        )
    elif submission == "silence":
        reply = await bridge.submit_silence()
    else:
        reply = await bridge.submit_section_time_barge_in(
            partial_candidate_text="  partial   answer  ",
        )

    command = graph.ainvoke.await_args.args[0]
    event = command.resume
    assert reply == "next reply"
    assert graph.ainvoke.await_args.kwargs["config"] == {
        "configurable": {"thread_id": "candidate-assessment-id"},
        "run_name": "iBot interview turn",
        "tags": ["ibot", "livekit-interview", "development"],
        "metadata": {
            "environment": "development",
            "candidate_assessment_id": "candidate-assessment-id",
            "interview_session_id": "interview-session-id",
        },
    }
    assert {key: event[key] for key in expected_event} == expected_event
    assert event["elapsed_secs"] == 12
    assert event["received_at"]
    assert bridge.state["timer_started"] is True
    assert bridge.state["timer_started_at"] == "2026-01-01T00:00:00+00:00"
    get_graph.assert_awaited_once_with()
