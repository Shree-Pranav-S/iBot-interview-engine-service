"""
WebSocket endpoint for real-time practice/demo interview sessions.
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.core.services.demo_bot_service import DemoBotService
from src.core.services.stt_service import STTService
from src.core.services.tts_service import synthesize_speech
from src.schemas.ws_messages import (
    ClientMessage,
    ClientMessageType,
    ServerMessageType,
)
from src.utils.ws import _server_msg

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/interview/demo")
async def demo_interview_websocket(websocket: WebSocket) -> None:
    """Accept and orchestrate a real-time practice AI interview session."""
    candidate_id: str = websocket.headers.get("x-candidate-id", "unknown")
    assessment_id: str = websocket.headers.get("x-assessment-id", "unknown")

    await websocket.accept()
    logger.info(
        "Demo interview session started: candidate=%s assessment=%s",
        candidate_id,
        assessment_id,
    )

    # Use DemoBotService for static responses
    demo_bot = DemoBotService(candidate_id=candidate_id, assessment_id=assessment_id)

    # Queue for server → client messages
    outbound_queue: asyncio.Queue[str | bytes | None] = asyncio.Queue()

    # Acknowledge connection
    await outbound_queue.put(
        _server_msg(
            ServerMessageType.CONNECTION_ACK,
            session_id=f"demo:{candidate_id}:{assessment_id}",
            message="Practice session connected! Speak into your microphone.",
        )
    )

    # ── STT integration: audio chunks → transcripts → DemoBot ─────────────────

    async def audio_pipeline() -> None:
        """Drain audio queue through Deepgram and feed transcripts to DemoBot."""
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

        async def drain_audio(stt: STTService) -> None:
            while True:
                chunk = await audio_queue.get()
                if chunk is None:
                    break
                await stt.send_audio(chunk)

        async def process_transcripts(stt: STTService) -> None:
            accumulated_text: list[str] = []
            timer_task: asyncio.Task | None = None

            async def wait_and_trigger(delay: float = 3.0) -> None:
                try:
                    if delay > 0:
                        await asyncio.sleep(delay)
                    if not accumulated_text:
                        return
                    combined_text = " ".join(accumulated_text).strip()
                    accumulated_text.clear()
                    if not combined_text:
                        return

                    # Commit the accumulated text to a single final box on the frontend
                    await outbound_queue.put(
                        _server_msg(
                            ServerMessageType.FINAL_TRANSCRIPT,
                            text=combined_text,
                        )
                    )

                    try:
                        # Use DemoBot static response
                        reply = await demo_bot.get_reply(combined_text)
                        await outbound_queue.put(
                            _server_msg(
                                ServerMessageType.ASSISTANT_TEXT,
                                text=reply,
                            )
                        )
                        # Generate TTS
                        try:
                            audio_bytes = await synthesize_speech(reply)
                            await outbound_queue.put(
                                _server_msg(
                                    ServerMessageType.TTS_AUDIO,
                                    encoding="mp3",
                                    size=len(audio_bytes),
                                )
                            )
                            await outbound_queue.put(audio_bytes)
                        except Exception:
                            logger.exception("TTS synthesis failed in demo")
                    except Exception:
                        logger.exception("Demo bot reply generation failed")
                        await outbound_queue.put(
                            _server_msg(
                                ServerMessageType.ERROR,
                                message="Demo bot response error.",
                            )
                        )
                except asyncio.CancelledError:
                    pass

            try:
                async for event in stt.transcript_events():
                    if timer_task and not timer_task.done():
                        timer_task.cancel()

                    text = event.text.strip()

                    if not event.is_final:
                        display_text = " ".join([*accumulated_text, text]).strip()
                        if display_text:
                            await outbound_queue.put(
                                _server_msg(
                                    ServerMessageType.PARTIAL_TRANSCRIPT,
                                    text=display_text,
                                )
                            )
                        continue

                    # Final events: segment_final, speech_final, or utterance_end
                    if text:
                        accumulated_text.append(text)

                    # Keep the UI box partial until the turn is explicitly processed
                    display_text = " ".join(accumulated_text).strip()
                    if display_text:
                        await outbound_queue.put(
                            _server_msg(
                                ServerMessageType.PARTIAL_TRANSCRIPT, text=display_text
                            )
                        )

                    event_type = getattr(event, "event_type", "")
                    if event_type == "utterance_end":
                        timer_task = asyncio.create_task(wait_and_trigger(0.2))
                    else:
                        timer_task = asyncio.create_task(wait_and_trigger(2.5))
            finally:
                if timer_task and not timer_task.done():
                    timer_task.cancel()

        # Expose the audio_queue via shared state
        session_state["audio_queue"] = audio_queue

        async with STTService() as stt:
            await asyncio.gather(
                drain_audio(stt),
                process_transcripts(stt),
            )

    session_state: dict = {}
    audio_pipeline_task: asyncio.Task | None = None

    # ── inbound task ──────────────────────────────────────────────────────────

    async def inbound_task() -> None:
        nonlocal audio_pipeline_task

        while True:
            try:
                frame = await websocket.receive()
            except WebSocketDisconnect:
                logger.info("Demo client disconnected: candidate=%s", candidate_id)
                break

            if frame.get("type") == "websocket.disconnect":
                break

            # Audio chunks
            if "bytes" in frame and frame["bytes"]:
                aq = session_state.get("audio_queue")
                if aq is not None:
                    await aq.put(frame["bytes"])
                continue

            if "text" not in frame or not frame["text"]:
                continue

            # Control messages
            try:
                data = json.loads(frame["text"])
                msg = ClientMessage.model_validate(data)
            except Exception:
                await outbound_queue.put(
                    _server_msg(
                        ServerMessageType.ERROR, message="Invalid message format."
                    )
                )
                continue

            if msg.type == ClientMessageType.PING:
                await outbound_queue.put(_server_msg(ServerMessageType.PONG))

            elif msg.type == ClientMessageType.SESSION_START:
                if audio_pipeline_task is None or audio_pipeline_task.done():
                    audio_pipeline_task = asyncio.create_task(audio_pipeline())

                # Send initial demo greeting
                try:
                    greeting = (
                        "Welcome to your demo practice session. I will act as the interviewer. "
                        "Let's begin. Can you introduce yourself and tell me what role you are applying for?"
                    )
                    await outbound_queue.put(
                        _server_msg(ServerMessageType.ASSISTANT_TEXT, text=greeting)
                    )
                    try:
                        audio_bytes = await synthesize_speech(greeting)
                        await outbound_queue.put(
                            _server_msg(
                                ServerMessageType.TTS_AUDIO,
                                encoding="mp3",
                                size=len(audio_bytes),
                            )
                        )
                        await outbound_queue.put(audio_bytes)
                    except Exception:
                        logger.exception("TTS greeting synthesis failed in demo")
                except Exception:
                    logger.exception("Demo bot initial message failed")

            elif msg.type == ClientMessageType.STOP:
                logger.info("Demo stop received from candidate=%s", candidate_id)
                break

            elif msg.type == ClientMessageType.TEXT_MESSAGE:
                text = msg.payload.get("text", "").strip()
                if text:
                    try:
                        reply = await demo_bot.get_reply(text)
                        await outbound_queue.put(
                            _server_msg(ServerMessageType.ASSISTANT_TEXT, text=reply)
                        )
                        try:
                            audio_bytes = await synthesize_speech(reply)
                            await outbound_queue.put(
                                _server_msg(
                                    ServerMessageType.TTS_AUDIO,
                                    encoding="mp3",
                                    size=len(audio_bytes),
                                )
                            )
                            await outbound_queue.put(audio_bytes)
                        except Exception:
                            logger.exception(
                                "TTS synthesis failed in demo text message"
                            )
                    except Exception:
                        logger.exception("Demo bot reply generation failed for text")
                        await outbound_queue.put(
                            _server_msg(
                                ServerMessageType.ERROR,
                                message="Demo bot response error.",
                            )
                        )

        # Cleanup
        await outbound_queue.put(None)
        aq = session_state.get("audio_queue")
        if aq is not None:
            await aq.put(None)

    # ── outbound task ─────────────────────────────────────────────────────────

    async def outbound_task() -> None:
        while True:
            item = await outbound_queue.get()
            if item is None:
                break
            try:
                if isinstance(item, bytes):
                    await websocket.send_bytes(item)
                else:
                    await websocket.send_text(item)
            except WebSocketDisconnect:
                break
            except Exception:
                logger.exception("Error sending demo frame to client")

    # ── run concurrently ──────────────────────────────────────────────────────

    tasks = {
        asyncio.create_task(inbound_task(), name="inbound"),
        asyncio.create_task(outbound_task(), name="outbound"),
    }

    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

    for task in pending:
        task.cancel()

    if audio_pipeline_task and not audio_pipeline_task.done():
        audio_pipeline_task.cancel()

    for task in done:
        try:
            task.result()
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except Exception:
            logger.exception("Unhandled error in demo interview task")

    logger.info(
        "Demo interview session closed: candidate=%s assessment=%s",
        candidate_id,
        assessment_id,
    )
