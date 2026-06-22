"""WebSocket utility helpers for interview sessions."""

import asyncio
import logging

from src.core.services.tts_service import synthesize_speech
from src.schemas.ws_messages import (
    ServerMessage,
    ServerMessageType,
)

logger = logging.getLogger(__name__)


def _server_msg(msg_type: ServerMessageType, **kwargs: object) -> str:
    return ServerMessage(type=msg_type, payload=dict(kwargs)).to_json()


def _estimate_spoken_duration_secs(text: str) -> float:
    """Estimate how long the TTS audio will take to play."""
    words = len(text.split())
    if words == 0:
        return 0.0
    return min(45.0, max(1.2, words / 2.55 + 0.7))


async def _send_bot_reply(
    outbound_queue: asyncio.Queue[str | bytes | None],
    text: str,
    reply_type: str,
    turn_number: int = 0,
    bot_speaking_event: asyncio.Event | None = None,
) -> None:
    """Send bot text + TTS audio to the client.  **Non-blocking.**

    Clears ``bot_speaking_event`` immediately, sends the text/audio frames,
    then launches a background task that waits for the estimated playback
    duration and *then* sets the event + sends BOT_DONE_SPEAKING.

    This is intentionally non-blocking so that the caller (and the
    inbound_task loop) is never stalled.  The ``bot_speaking_event``
    gate in ``process_transcript`` and ``wait_and_trigger`` prevents
    any candidate response from being committed while the bot is still
    playing audio.
    """
    if not text:
        return

    estimated_duration_secs = _estimate_spoken_duration_secs(text)

    # Signal: bot is now speaking — cleared *before* anything is sent
    # so every downstream guard sees the "bot talking" state immediately.
    if bot_speaking_event is not None:
        # Cancel any previous done-task so we don't get a stale .set()
        prev = getattr(bot_speaking_event, "_bot_done_task", None)
        if prev is not None and not prev.done():
            prev.cancel()
        bot_speaking_event.clear()

    # 1. Subtitle text
    await outbound_queue.put(
        _server_msg(ServerMessageType.ASSISTANT_TEXT, text=text, reply_type=reply_type)
    )

    # 2. BOT_SPEAKING event (so frontend can mute mic visualisation etc.)
    await outbound_queue.put(
        _server_msg(
            ServerMessageType.BOT_SPEAKING,
            text=text,
            turn_number=turn_number,
            estimated_duration_ms=int(estimated_duration_secs * 1000),
        )
    )

    # 3. Synthesize and send TTS audio
    try:
        audio_bytes = await synthesize_speech(text)
        await outbound_queue.put(
            _server_msg(
                ServerMessageType.TTS_AUDIO,
                encoding="mp3",
                size=len(audio_bytes),
                estimated_duration_ms=int(estimated_duration_secs * 1000),
            )
        )
        await outbound_queue.put(audio_bytes)
    except Exception:
        logger.exception("TTS synthesis failed")

    # 4. Fire-and-forget: wait for playback, then signal done
    async def _mark_done() -> None:
        try:
            if estimated_duration_secs > 0:
                await asyncio.sleep(estimated_duration_secs)
            await outbound_queue.put(_server_msg(ServerMessageType.BOT_DONE_SPEAKING))
        except asyncio.CancelledError:
            pass
        finally:
            if bot_speaking_event is not None:
                bot_speaking_event.set()

    task = asyncio.create_task(_mark_done(), name="bot-done-timer")
    if bot_speaking_event is not None:
        bot_speaking_event._bot_done_task = task  # type: ignore[attr-defined]


async def _send_section_info(
    outbound_queue: asyncio.Queue[str | bytes | None],
    state: dict,
) -> None:
    """Send section start/progress info from graph state."""
    sections = state.get("sections") or []
    section_idx = state.get("current_section_index", 0)

    if section_idx < len(sections):
        section = sections[section_idx]
        await outbound_queue.put(
            _server_msg(
                ServerMessageType.SECTION_START,
                section_name=section["name"],
                skill=section["skill"],
                time_budget_secs=section["time_budget_secs"],
                section_number=section_idx + 1,
                total_sections=len(sections),
            )
        )


def _is_technical_section(skill: str, section_name: str) -> bool:
    """Determine if the current section/skill is technical."""
    non_tech_indicators = {
        "communication",
        "behavioural",
        "behavioral",
        "closing",
        "general",
        "cultural",
        "culture",
        "self introduction",
        "intro",
        "introduction",
    }
    skill_lower = skill.lower()
    name_lower = section_name.lower()

    return not any(
        indicator in skill_lower or indicator in name_lower
        for indicator in non_tech_indicators
    )
