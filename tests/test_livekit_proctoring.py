"""LiveKit proctoring packet validation and termination behavior."""

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from src.control.agents.livekit_agent import InterviewLiveKitAgent
from src.utils.livekit import (
    FACE_ABSENT_EVENT,
    INTERVIEW_PROCTORING_TOPIC,
    MULTIPLE_FACES_EVENT,
    TAB_SWITCH_EVENT,
)


def _face_packet_payload(
    *,
    event_type: str = FACE_ABSENT_EVENT,
    duration_ms: int = 5_000,
) -> dict[str, object]:
    return {
        "type": event_type,
        "event_id": str(uuid4()),
        "condition_started_at": datetime.now(UTC).isoformat(),
        "observed_duration_ms": duration_ms,
        "sample_count": 11,
        "max_face_count": 0 if event_type == FACE_ABSENT_EVENT else 2,
        "min_confidence": 0.72,
        "max_confidence": 0.93,
        "source": "mediapipe_face_detector",
        "detector_version": "mediapipe-tasks-vision@0.10.35",
        "model_name": "blaze_face_short_range_float16",
    }


async def test_fifth_tab_switch_stops_audio_and_queues_terminated_evaluation() -> None:
    agent = cast(Any, object.__new__(InterviewLiveKitAgent))
    agent._interview_closed = False
    agent._closing_finalize_requested = False
    agent.bridge = SimpleNamespace(
        candidate_assessment_id=str(uuid4()),
        record_tab_switch=AsyncMock(
            return_value={
                "appended": False,
                "recorded": True,
                "tab_switch_count": 5,
                "terminated": True,
                "termination_reason": "tab_switch_limit",
                "status": "TERMINATED",
            }
        ),
        finalize_policy_termination=AsyncMock(),
    )
    agent._cancel_pending_silence = Mock()
    agent._cancel_pending_section_barge = Mock()
    agent._publish_tab_switch_recorded_signal = AsyncMock()
    agent._publish_interview_terminated_signal = AsyncMock()
    session = SimpleNamespace(
        input=SimpleNamespace(set_audio_enabled=Mock()),
        interrupt=AsyncMock(),
        shutdown=Mock(),
    )
    agent._get_activity_or_raise = Mock(return_value=SimpleNamespace(session=session))
    event_id = uuid4()
    packet = SimpleNamespace(
        topic=INTERVIEW_PROCTORING_TOPIC,
        participant=SimpleNamespace(identity="candidate-123"),
        data=json.dumps({"type": TAB_SWITCH_EVENT, "event_id": str(event_id)}).encode(),
    )

    await agent.handle_proctoring_packet(packet)

    agent.bridge.record_tab_switch.assert_awaited_once()
    assert agent._interview_closed is True
    assert agent._closing_finalize_requested is True
    session.input.set_audio_enabled.assert_called_once_with(False)
    session.interrupt.assert_awaited_once_with(force=True)
    agent._publish_tab_switch_recorded_signal.assert_awaited_once_with(
        event_id=event_id,
        tab_switch_count=5,
        appended=False,
        terminated=True,
    )
    agent._publish_interview_terminated_signal.assert_awaited_once_with(
        reason="tab_switch_limit",
        tab_switch_count=5,
    )
    agent.bridge.finalize_policy_termination.assert_awaited_once_with(
        "tab_switch_limit"
    )
    session.shutdown.assert_called_once_with(drain=True)


async def test_duplicate_tab_switch_is_acknowledged_with_authoritative_count() -> None:
    agent = cast(Any, object.__new__(InterviewLiveKitAgent))
    agent._interview_closed = False
    agent.bridge = SimpleNamespace(
        candidate_assessment_id=str(uuid4()),
        record_tab_switch=AsyncMock(
            return_value={
                "appended": False,
                "tab_switch_count": 3,
                "terminated": False,
                "status": "IN_PROGRESS",
            }
        ),
    )
    agent._publish_tab_switch_recorded_signal = AsyncMock()
    agent._publish_interview_terminated_signal = AsyncMock()
    event_id = uuid4()
    packet = SimpleNamespace(
        topic=INTERVIEW_PROCTORING_TOPIC,
        participant=SimpleNamespace(identity="candidate-123"),
        data=json.dumps({"type": TAB_SWITCH_EVENT, "event_id": str(event_id)}).encode(),
    )

    await agent.handle_proctoring_packet(packet)

    agent._publish_tab_switch_recorded_signal.assert_awaited_once_with(
        event_id=event_id,
        tab_switch_count=3,
        appended=False,
        terminated=False,
    )
    agent._publish_interview_terminated_signal.assert_not_awaited()
    assert agent._interview_closed is False


async def test_proctoring_waits_for_opening_context_before_persisting() -> None:
    agent = cast(Any, object.__new__(InterviewLiveKitAgent))
    agent._interview_closed = False
    agent._proctoring_ready = asyncio.Event()
    agent.bridge = SimpleNamespace(
        candidate_assessment_id=str(uuid4()),
        record_tab_switch=AsyncMock(
            return_value={
                "appended": True,
                "tab_switch_count": 1,
                "terminated": False,
                "status": "IN_PROGRESS",
            }
        ),
    )
    agent._publish_tab_switch_recorded_signal = AsyncMock()
    event_id = uuid4()
    packet = SimpleNamespace(
        topic=INTERVIEW_PROCTORING_TOPIC,
        participant=SimpleNamespace(identity="candidate-123"),
        data=json.dumps({"type": TAB_SWITCH_EVENT, "event_id": str(event_id)}).encode(),
    )

    pending = asyncio.create_task(agent.handle_proctoring_packet(packet))
    await asyncio.sleep(0)
    agent.bridge.record_tab_switch.assert_not_awaited()

    agent._proctoring_ready.set()
    await pending

    agent.bridge.record_tab_switch.assert_awaited_once()
    agent._publish_tab_switch_recorded_signal.assert_awaited_once()


async def test_non_candidate_or_wrong_topic_packet_is_ignored() -> None:
    agent = cast(Any, object.__new__(InterviewLiveKitAgent))
    agent._interview_closed = False
    agent.bridge = SimpleNamespace(
        candidate_assessment_id=str(uuid4()),
        record_tab_switch=AsyncMock(),
    )
    packet = SimpleNamespace(
        topic="some.other.topic",
        participant=SimpleNamespace(identity="interview-bot-123"),
        data=b"{}",
    )

    await agent.handle_proctoring_packet(packet)

    agent.bridge.record_tab_switch.assert_not_awaited()


async def test_valid_multiple_faces_event_is_persisted_and_acknowledged() -> None:
    agent = cast(Any, object.__new__(InterviewLiveKitAgent))
    agent._interview_closed = False
    agent.bridge = SimpleNamespace(
        candidate_assessment_id=str(uuid4()),
        record_proctoring_event=AsyncMock(
            return_value={"appended": True, "status": "IN_PROGRESS"}
        ),
    )
    agent._publish_proctoring_event_recorded_signal = AsyncMock()
    payload = _face_packet_payload(
        event_type=MULTIPLE_FACES_EVENT,
        duration_ms=3_000,
    )
    packet = SimpleNamespace(
        topic=INTERVIEW_PROCTORING_TOPIC,
        participant=SimpleNamespace(identity="candidate-123"),
        data=json.dumps(payload).encode(),
    )

    await agent.handle_proctoring_packet(packet)

    agent.bridge.record_proctoring_event.assert_awaited_once()
    call = agent.bridge.record_proctoring_event.await_args.kwargs
    assert call["event_type"] == MULTIPLE_FACES_EVENT
    assert call["observed_duration_ms"] == 3_000
    agent._publish_proctoring_event_recorded_signal.assert_awaited_once_with(
        event_id=call["event_id"],
        appended=True,
        terminated=False,
    )


async def test_continuous_face_absence_terminates_and_queues_evaluation() -> None:
    agent = cast(Any, object.__new__(InterviewLiveKitAgent))
    agent._interview_closed = False
    agent._closing_finalize_requested = False
    agent.bridge = SimpleNamespace(
        candidate_assessment_id=str(uuid4()),
        record_proctoring_event=AsyncMock(
            return_value={
                "appended": False,
                "recorded": True,
                "terminated": True,
                "termination_reason": "face_absent_continuous_duration_exceeded",
                "status": "TERMINATED",
            }
        ),
        finalize_policy_termination=AsyncMock(),
    )
    agent._cancel_pending_silence = Mock()
    agent._cancel_pending_section_barge = Mock()
    agent._publish_proctoring_event_recorded_signal = AsyncMock()
    agent._publish_interview_terminated_signal = AsyncMock()
    session = SimpleNamespace(
        input=SimpleNamespace(set_audio_enabled=Mock()),
        interrupt=AsyncMock(),
        shutdown=Mock(),
    )
    agent._get_activity_or_raise = Mock(return_value=SimpleNamespace(session=session))
    payload = _face_packet_payload(duration_ms=30_000)
    packet = SimpleNamespace(
        topic=INTERVIEW_PROCTORING_TOPIC,
        participant=SimpleNamespace(identity="candidate-123"),
        data=json.dumps(payload).encode(),
    )

    await agent.handle_proctoring_packet(packet)

    event_id = agent.bridge.record_proctoring_event.await_args.kwargs["event_id"]
    agent._publish_proctoring_event_recorded_signal.assert_awaited_once_with(
        event_id=event_id,
        appended=False,
        terminated=True,
    )
    agent._publish_interview_terminated_signal.assert_awaited_once_with(
        reason="face_absent_continuous_duration_exceeded",
        tab_switch_count=None,
    )
    agent.bridge.finalize_policy_termination.assert_awaited_once_with(
        "face_absent_continuous_duration_exceeded"
    )
    assert agent._interview_closed is True
    assert agent._closing_finalize_requested is True
    session.input.set_audio_enabled.assert_called_once_with(False)
    session.interrupt.assert_awaited_once_with(force=True)
    session.shutdown.assert_called_once_with(drain=True)


async def test_continuous_multiple_faces_terminates_at_twenty_seconds() -> None:
    agent = cast(Any, object.__new__(InterviewLiveKitAgent))
    agent._interview_closed = False
    agent._closing_finalize_requested = False
    agent.bridge = SimpleNamespace(
        candidate_assessment_id=str(uuid4()),
        record_proctoring_event=AsyncMock(
            return_value={
                "appended": False,
                "recorded": True,
                "terminated": True,
                "termination_reason": "multiple_faces_continuous_duration_exceeded",
                "status": "TERMINATED",
            }
        ),
        finalize_policy_termination=AsyncMock(),
    )
    agent._cancel_pending_silence = Mock()
    agent._cancel_pending_section_barge = Mock()
    agent._publish_proctoring_event_recorded_signal = AsyncMock()
    agent._publish_interview_terminated_signal = AsyncMock()
    session = SimpleNamespace(
        input=SimpleNamespace(set_audio_enabled=Mock()),
        interrupt=AsyncMock(),
        shutdown=Mock(),
    )
    agent._get_activity_or_raise = Mock(return_value=SimpleNamespace(session=session))
    packet = SimpleNamespace(
        topic=INTERVIEW_PROCTORING_TOPIC,
        participant=SimpleNamespace(identity="candidate-123"),
        data=json.dumps(
            _face_packet_payload(
                event_type=MULTIPLE_FACES_EVENT,
                duration_ms=20_000,
            )
        ).encode(),
    )

    await agent.handle_proctoring_packet(packet)

    agent._publish_interview_terminated_signal.assert_awaited_once_with(
        reason="multiple_faces_continuous_duration_exceeded",
        tab_switch_count=None,
    )
    agent.bridge.finalize_policy_termination.assert_awaited_once_with(
        "multiple_faces_continuous_duration_exceeded"
    )


async def test_face_event_below_duration_policy_is_ignored() -> None:
    agent = cast(Any, object.__new__(InterviewLiveKitAgent))
    agent._interview_closed = False
    agent.bridge = SimpleNamespace(
        candidate_assessment_id=str(uuid4()),
        record_proctoring_event=AsyncMock(),
    )
    agent._publish_proctoring_event_recorded_signal = AsyncMock()
    packet = SimpleNamespace(
        topic=INTERVIEW_PROCTORING_TOPIC,
        participant=SimpleNamespace(identity="candidate-123"),
        data=json.dumps(_face_packet_payload(duration_ms=4_999)).encode(),
    )

    await agent.handle_proctoring_packet(packet)

    agent.bridge.record_proctoring_event.assert_not_awaited()
    agent._publish_proctoring_event_recorded_signal.assert_not_awaited()
