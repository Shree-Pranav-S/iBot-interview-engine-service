"""
WebSocket endpoint for real-time AI interview sessions.

Flow
----
1.  Gateway validates the candidate's one-time token and injects
    X-Candidate-Id / X-Assessment-Id headers before forwarding here.
2.  We accept the connection and start two concurrent tasks:
      • inbound_task  – reads client frames (binary audio only)
      • outbound_task – pumps server messages from an async queue
3.  On SESSION_START, the orchestrator first checks for an existing
    checkpoint. If one exists and is resumable (within the grace period),
    the session resumes — the current question is re-sent via TTS.
    Otherwise, a fresh session is started: init → opening → await_response.
4.  Audio from the client microphone is streamed through Deepgram STT.
    When a complete utterance is detected, it's submitted to the graph
    via orchestrator.submit_response(transcript). The graph processes
    classify → evaluate → generate_question → await_response (interrupt)
    and the resulting bot reply is sent back as text + TTS audio.
5.  On disconnect, the session is paused with a grace period. If the
    candidate reconnects (new WebSocket with the same assessment_id),
    the session is resumed from the checkpoint.
6.  The microphone stream is always active from SESSION_START onwards;
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

from src.core.services.evaluation_service import run_holistic_evaluation
from src.core.services.interview_service import InterviewOrchestrator
from src.core.services.stt_service import STTService
from src.core.services.tts_service import synthesize_speech
from src.data.repositories.interview_repository import sync_interview_state
from src.schemas.ws_messages import (
    ClientMessage,
    ClientMessageType,
    ServerMessageType,
)
from src.utils.ws import _server_msg

logger = logging.getLogger(__name__)
router = APIRouter()


# ── helpers ───────────────────────────────────────────────────────────────────


async def _send_bot_reply(
    outbound_queue: asyncio.Queue[str | bytes | None],
    text: str,
    reply_type: str,
    turn_number: int = 0,
) -> None:
    """Send bot text + TTS audio to the client."""
    if not text:
        return

    # 1. Send the text for subtitles
    await outbound_queue.put(
        _server_msg(ServerMessageType.ASSISTANT_TEXT, text=text, reply_type=reply_type)
    )

    # 2. Signal bot is speaking
    await outbound_queue.put(
        _server_msg(ServerMessageType.BOT_SPEAKING, text=text, turn_number=turn_number)
    )

    # 3. Synthesize and send TTS audio
    try:
        audio_bytes = await synthesize_speech(text)
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

    # 4. Signal bot done speaking
    await outbound_queue.put(_server_msg(ServerMessageType.BOT_DONE_SPEAKING))


async def _send_section_info(
    outbound_queue: asyncio.Queue[str | bytes | None],
    state: dict,
) -> None:
    """Send section start/progress info from graph state."""
    sections = state.get("sections", [])
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


async def _handle_session_resume(
    orchestrator: InterviewOrchestrator,
    outbound_queue: asyncio.Queue[str | bytes | None],
) -> bool:
    """
    Attempt to resume an existing session from checkpoint.

    Returns True if the session was successfully resumed, False otherwise.
    On success, sends SESSION_RESUMED + section info + re-sends the
    current question via TTS so the candidate knows where they left off.
    """
    state = await orchestrator.resume_session()

    if state is None:
        return False

    # Session resumed! Tell the client.
    turn_number = state.get("turn_number", 0)
    section_idx = state.get("current_section_index", 0)
    sections = state.get("sections", [])
    current_section_name = (
        sections[section_idx]["name"] if section_idx < len(sections) else "unknown"
    )

    await outbound_queue.put(
        _server_msg(
            ServerMessageType.SESSION_RESUMED,
            turn_number=turn_number,
            section_index=section_idx,
            section_name=current_section_name,
            message=(
                "Welcome back! Your interview session has been restored. "
                "Let me repeat the last question."
            ),
        )
    )

    # Send section info
    await _send_section_info(outbound_queue, state)

    # Re-send the current question so the candidate knows where
    # they left off. The bot should sound natural about it.
    current_question = state.get("current_question_text", "")
    if current_question:
        resume_text = (
            f"Welcome back! Let me pick up where we left off. {current_question}"
        )
        await _send_bot_reply(
            outbound_queue,
            resume_text,
            "question",
            turn_number=turn_number,
        )

    logger.info(
        "Session resumed: assessment=%s turn=%d section=%s",
        orchestrator.assessment_id,
        turn_number,
        current_section_name,
    )

    return True


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

    # Per-session orchestrator
    compiled_graph = getattr(websocket.app.state, "compiled_graph", None)
    orchestrator = InterviewOrchestrator(
        candidate_id=candidate_id,
        assessment_id=assessment_id,
        compiled_graph=compiled_graph,
    )

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

    # ── STT integration: audio chunks → transcripts → Graph ───────────────────

    # Shared state
    session_state: dict = {}
    audio_pipeline_task: asyncio.Task | None = None
    processing_lock = asyncio.Lock()

    async def silence_watchdog() -> None:
        try:
            await asyncio.sleep(15.0)
            asyncio.create_task(_process_transcript("__SILENCE__"))
        except asyncio.CancelledError:
            pass

    async def timeout_watchdog(total_budget_secs: int, total_elapsed: int) -> None:
        try:
            remaining = total_budget_secs - total_elapsed
            if remaining > 0:
                await asyncio.sleep(remaining)
            asyncio.create_task(_process_transcript("__TIME_UP__"))
        except asyncio.CancelledError:
            pass

    async def _process_transcript(combined_text: str) -> None:
        """Submit the accumulated transcript to the LangGraph and relay the response."""
        if combined_text not in ("__SILENCE__", "__TIME_UP__"):
            wd = session_state.get("watchdog_task")
            if wd and not wd.done():
                wd.cancel()

        async with processing_lock:
            try:
                state = await orchestrator.submit_response(combined_text)

                # Extract bot reply from graph state
                bot_text = state.get("bot_reply_text", "")
                bot_type = state.get("bot_reply_type", "question")
                turn_number = state.get("turn_number", 0)

                # Send section info if it changed
                await _send_section_info(outbound_queue, state)

                # Send the bot reply
                await _send_bot_reply(outbound_queue, bot_text, bot_type, turn_number)

                # Check for session completion
                session_status = state.get("session_status", "in_progress")
                if session_status == "completed":
                    await outbound_queue.put(
                        _server_msg(ServerMessageType.INTERVIEW_COMPLETE)
                    )
                elif session_status == "terminated":
                    await outbound_queue.put(
                        _server_msg(ServerMessageType.SESSION_TERMINATED)
                    )

                # Sync state to Postgres asynchronously
                asyncio.create_task(sync_interview_state(state))

                # Trigger holistic evaluation if interview is over
                if session_status in ("completed", "terminated"):
                    asyncio.create_task(
                        run_holistic_evaluation(orchestrator.assessment_id)
                    )
                else:
                    session_state["watchdog_task"] = asyncio.create_task(
                        silence_watchdog()
                    )

            except Exception:
                logger.exception("Graph processing failed")
                await outbound_queue.put(
                    _server_msg(
                        ServerMessageType.ERROR,
                        message="AI response error. Please try again.",
                    )
                )

    async def audio_pipeline() -> None:
        """Drain the audio queue through Deepgram, then feed transcripts to Graph."""
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
                        await outbound_queue.put(
                            _server_msg(
                                ServerMessageType.FINAL_TRANSCRIPT,
                                text=combined_text,
                            )
                        )
                        await _process_transcript(combined_text)
                except asyncio.CancelledError:
                    pass

            try:
                async for event in stt.transcript_events():
                    if event.text.strip():
                        wd = session_state.get("watchdog_task")
                        if wd and not wd.done():
                            wd.cancel()
                        session_state["watchdog_task"] = asyncio.create_task(
                            silence_watchdog()
                        )

                    if timer_task and not timer_task.done():
                        timer_task.cancel()

                    if not event.is_final:
                        current_partial = event.text
                        display_text = " ".join(
                            accumulated_text + [current_partial]
                        ).strip()
                        await outbound_queue.put(
                            _server_msg(
                                ServerMessageType.PARTIAL_TRANSCRIPT,
                                text=display_text,
                            )
                        )
                    else:
                        if event.text.strip():
                            accumulated_text.append(event.text.strip())
                            display_text = " ".join(accumulated_text).strip()
                            await outbound_queue.put(
                                _server_msg(
                                    ServerMessageType.PARTIAL_TRANSCRIPT,
                                    text=display_text,
                                )
                            )
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

            elif msg.type in (
                ClientMessageType.SESSION_START,
                ClientMessageType.SESSION_RESUME,
            ):
                # Start the always-on audio pipeline
                if audio_pipeline_task is None or audio_pipeline_task.done():
                    audio_pipeline_task = asyncio.create_task(audio_pipeline())

                # ── Reconnection: try to resume from checkpoint first ─────
                try:
                    resumed = await _handle_session_resume(orchestrator, outbound_queue)

                    if not resumed:
                        # No valid checkpoint → start fresh
                        state = await orchestrator.start_session()

                        # Send section info
                        await _send_section_info(outbound_queue, state)

                        # Send the opening monologue
                        opening_text = state.get("bot_reply_text", "")
                        if opening_text:
                            await _send_bot_reply(
                                outbound_queue,
                                opening_text,
                                "opening",
                                turn_number=0,
                            )

                        # Sync initial state to Postgres
                        asyncio.create_task(sync_interview_state(state))

                    # start timeout watchdog
                    current_state = await orchestrator.get_current_state()
                    if current_state and "timeout_task" not in session_state:
                        plan = current_state.get("interview_plan", {})
                        total_budget_secs = plan.get("total_mins", 0) * 60
                        total_elapsed = current_state.get("total_elapsed_secs", 0)
                        if total_budget_secs > 0:
                            session_state["timeout_task"] = asyncio.create_task(
                                timeout_watchdog(total_budget_secs, total_elapsed)
                            )

                    session_state["watchdog_task"] = asyncio.create_task(
                        silence_watchdog()
                    )

                except Exception:
                    logger.exception("Failed to start/resume interview session")
                    await outbound_queue.put(
                        _server_msg(
                            ServerMessageType.ERROR,
                            message="Failed to initialise interview. Please try again.",
                        )
                    )

            elif msg.type == ClientMessageType.TEXT_MESSAGE:
                # Handle typed text input as if it were a transcript
                text = msg.payload.get("text", "").strip()
                if text:
                    await _process_transcript(text)

            elif msg.type == ClientMessageType.STOP:
                logger.info("Stop received from candidate=%s", candidate_id)
                break

        # ── Disconnect cleanup ────────────────────────────────────────────────
        # Pause the session for potential reconnection (grace period)
        try:
            current_state = await orchestrator.get_current_state()
            if current_state:
                status = current_state.get("session_status", "")
                if status == "in_progress":
                    await orchestrator.pause_session()
                    # Sync paused state to Postgres
                    paused_state = await orchestrator.get_current_state()
                    if paused_state:
                        asyncio.create_task(sync_interview_state(paused_state))

                    logger.info(
                        "Session paused for reconnection: assessment=%s",
                        assessment_id,
                    )
        except Exception:
            logger.exception("Failed to pause session on disconnect")

        # Signal outbound to finish and stop audio pipeline
        await outbound_queue.put(None)
        aq = session_state.get("audio_queue")
        if aq is not None:
            await aq.put(None)

        wd = session_state.get("watchdog_task")
        if wd and not wd.done():
            wd.cancel()

        td = session_state.get("timeout_task")
        if td and not td.done():
            td.cancel()

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
