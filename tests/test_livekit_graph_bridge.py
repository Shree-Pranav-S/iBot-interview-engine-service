"""Behavior contracts for the LiveKit-to-LangGraph bridge."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

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


async def test_policy_termination_flushes_and_queues_without_completing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate_id = str(uuid4())
    session_id = str(uuid4())
    drain = AsyncMock()
    trigger = AsyncMock(
        return_value={
            "holistic_evaluation_status": "QUEUED",
            "holistic_evaluation_task_id": "task-123",
            "next_action": "end",
        }
    )
    record_event = AsyncMock()
    persist_turn = AsyncMock()
    monkeypatch.setattr(bridge_module, "drain_background_persistence", drain)
    monkeypatch.setattr(bridge_module, "trigger_final_evaluation", trigger)
    monkeypatch.setattr(
        bridge_module,
        "try_record_event_in_background",
        record_event,
    )
    monkeypatch.setattr(
        bridge_module,
        "get_core_api_client",
        lambda: SimpleNamespace(persist_turn=persist_turn),
    )

    bridge = LiveKitInterviewBridge(
        candidate_assessment_id=candidate_id,
        interview_session_id=session_id,
        connection_id="connection-id",
    )
    bridge.state = {
        "candidate_assessment_id": candidate_id,
        "interview_session_id": session_id,
        "session_status": "TERMINATED",
        "elapsed_secs": 17,
    }

    await bridge.finalize_policy_termination("face_absent_continuous_duration_exceeded")

    drain.assert_awaited_once_with(session_id)
    persist_turn.assert_awaited_once_with(
        session_id=session_id,
        transcript_items=[],
        violations=[],
        elapsed_secs=17,
    )
    trigger.assert_awaited_once()
    queued_state = trigger.await_args.args[0]
    assert queued_state["session_status"] == "TERMINATED"
    assert queued_state["termination_reason"] == (
        "face_absent_continuous_duration_exceeded"
    )
    assert bridge.state["holistic_evaluation_status"] == "QUEUED"
    assert bridge.state["holistic_evaluation_task_id"] == "task-123"
    record_event.assert_awaited_once()
