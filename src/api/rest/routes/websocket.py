"""
WebSocket endpoint for real-time AI interview sessions.
Flow
----
1.  Gateway validates the candidate's one-time token and injects
    X-Candidate-Id / X-Assessment-Id headers before forwarding here.
2.  We accept the connection and start two concurrent tasks:
      • inbound_task  – reads client frames (binary audio only)
      • outbound_task – pumps server messages from an async queue
3.  Audio arrives from the client microphone, is streamed through the
    STT pipeline, and the final transcript is fed to the LLM. The reply
    is sent back as "assistant_text". TTS audio follows as a binary
    frame prefixed by a "tts_audio" notification text frame.
4.  The microphone stream is always active from SESSION_START onwards;
    there is no toggle. Either task cancelling (disconnect / stop command)
    tears down the other task and closes everything cleanly.

Header contract (set by the gateway WS proxy):
    X-Candidate-Id   : UUID of the authenticated candidate
    X-Assessment-Id  : UUID of the current assessment
    X-Internal-Service: "gateway"
"""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.core.services.llm_service import LLMService
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


# ── main handler ──────────────────────────────────────────────────────────────


@router.websocket("/ws/interview")
async def interview_websocket(websocket: WebSocket) -> None:
    """Accept and orchestrate a real-time AI interview session."""
    candidate_id: str = websocket.headers.get("x-candidate-id", "unknown")
    assessment_id: str = websocket.headers.get("x-assessment-id", "unknown")

    await websocket.accept()
    logger.info(
        "Interview session started: candidate=%s assessment=%s",
        candidate_id,
        assessment_id,
    )

    # Per-session services
    llm = LLMService(candidate_id=candidate_id, assessment_id=assessment_id)

    # Queue for server → client messages so outbound_task serialises all sends
    outbound_queue: asyncio.Queue[str | bytes | None] = asyncio.Queue()

    # Acknowledge the connection
    await outbound_queue.put(
        _server_msg(
            ServerMessageType.CONNECTION_ACK,
            session_id=f"{candidate_id}:{assessment_id}",
            message="Interview session established. Welcome to iBot!",
        )
    )

    # ── STT integration: audio chunks → transcripts → LLM ─────────────────────

    async def audio_pipeline() -> None:
        """Drain the audio queue through Deepgram, then feed transcripts to LLM."""
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

            async def wait_and_trigger() -> None:
                try:
                    await asyncio.sleep(3.0)
                    if accumulated_text:
                        combined_text = " ".join(accumulated_text)
                        accumulated_text.clear()
                        try:
                            reply = await llm.get_reply(combined_text)
                            await outbound_queue.put(
                                _server_msg(
                                    ServerMessageType.ASSISTANT_TEXT,
                                    text=reply,
                                )
                            )
                            # Generate TTS and enqueue binary audio
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
                                logger.exception("TTS synthesis failed")
                        except Exception:
                            logger.exception("LLM call failed after transcript")
                            await outbound_queue.put(
                                _server_msg(
                                    ServerMessageType.ERROR,
                                    message="AI response error. Please try again.",
                                )
                            )
                except asyncio.CancelledError:
                    pass

            try:
                async for event in stt.transcript_events():
                    if timer_task and not timer_task.done():
                        timer_task.cancel()

                    if not event.is_final:
                        await outbound_queue.put(
                            _server_msg(
                                ServerMessageType.PARTIAL_TRANSCRIPT,
                                text=event.text,
                            )
                        )
                    else:
                        await outbound_queue.put(
                            _server_msg(
                                ServerMessageType.FINAL_TRANSCRIPT,
                                text=event.text,
                            )
                        )
                        if event.text.strip():
                            accumulated_text.append(event.text.strip())
                            timer_task = asyncio.create_task(wait_and_trigger())
            finally:
                if timer_task and not timer_task.done():
                    timer_task.cancel()

        # Expose the audio_queue to inbound_task via shared session state
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
                logger.info("Client disconnected: candidate=%s", candidate_id)
                break

            if frame.get("type") == "websocket.disconnect":
                break

            # Binary frame — raw audio from the always-on microphone
            if "bytes" in frame and frame["bytes"]:
                aq = session_state.get("audio_queue")
                if aq is not None:
                    await aq.put(frame["bytes"])
                continue

            if "text" not in frame or not frame["text"]:
                continue

            # Parse control message
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
                # Start the always-on audio pipeline
                if audio_pipeline_task is None or audio_pipeline_task.done():
                    audio_pipeline_task = asyncio.create_task(audio_pipeline())

                # Send a greeting from the LLM
                try:
                    greeting = await llm.get_reply(
                        "Hello, I'm ready to start the interview."
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
                        logger.exception("TTS synthesis failed for greeting")
                except Exception:
                    logger.exception("LLM greeting failed")

            elif msg.type == ClientMessageType.STOP:
                logger.info("Stop received from candidate=%s", candidate_id)
                break

        # Signal outbound to finish and stop audio pipeline
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
                logger.exception("Error sending frame to client")

    # ── run both tasks concurrently ───────────────────────────────────────────

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
            logger.exception("Unhandled error in interview session task")

    logger.info(
        "Interview session closed: candidate=%s assessment=%s",
        candidate_id,
        assessment_id,
    )
