"""LiveKit proctoring packet validation and termination behavior."""

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from src.control.agents.livekit_agent import InterviewLiveKitAgent
from src.utils.livekit import INTERVIEW_PROCTORING_TOPIC, TAB_SWITCH_EVENT


async def test_sixth_tab_switch_stops_audio_without_running_completion() -> None:
    agent = cast(Any, object.__new__(InterviewLiveKitAgent))
    agent._interview_closed = False
    agent._closing_finalize_requested = False
    agent.bridge = SimpleNamespace(
        candidate_assessment_id=str(uuid4()),
        record_tab_switch=AsyncMock(
            return_value={
                "appended": True,
                "tab_switch_count": 6,
                "terminated": True,
                "status": "TERMINATED",
            }
        ),
    )
    agent._cancel_pending_silence = Mock()
    agent._cancel_pending_section_barge = Mock()
    agent._publish_tab_switch_recorded_signal = AsyncMock()
    agent._publish_interview_terminated_signal = AsyncMock()
    session = SimpleNamespace(
        input=SimpleNamespace(set_audio_enabled=Mock()),
        interrupt=AsyncMock(),
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
        tab_switch_count=6,
        appended=True,
        terminated=True,
    )
    agent._publish_interview_terminated_signal.assert_awaited_once_with(6)


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
