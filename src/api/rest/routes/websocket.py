"""WebSocket endpoint for real-time AI interview sessions."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.control.agents.state import InterviewState
from src.core.services.interview_service import InterviewOrchestrator
from src.core.services.stt_service import STTService
from src.data.repositories import interview_workflow_repository as db
from src.schemas.ws_messages import ClientMessage, ClientMessageType, ServerMessageType
from src.utils.ws import _send_bot_reply, _send_section_info, _server_msg

logger = logging.getLogger(__name__)
router = APIRouter()


def _session_closed(state: InterviewState | None) -> bool:
    if not state:
        return False
    return bool(state.get("closing_done") or state.get("should_close"))


def _new_bot_turns(
    previous_state: InterviewState | None,
    state: InterviewState,
) -> list[dict]:
    transcript = state.get("transcript") or []

    if previous_state is None and state.get("resumed"):
        bot_turns = [
            turn
            for turn in transcript
            if turn.get("speaker") == "bot" and turn.get("text")
        ]
        return bot_turns[-1:]

    previous_len = len(previous_state.get("transcript", [])) if previous_state else 0

    return [
        turn
        for turn in transcript[previous_len:]
        if turn.get("speaker") == "bot" and turn.get("text")
    ]


async def _send_graph_outputs(
    outbound_queue: asyncio.Queue[str | bytes | None],
    previous_state: InterviewState | None,
    state: InterviewState,
    bot_speaking_event: asyncio.Event,
) -> None:
    old_idx = previous_state.get("current_section_index") if previous_state else None
    new_idx = state.get("current_section_index")

    if previous_state is None or old_idx != new_idx:
        if previous_state is not None:
            await outbound_queue.put(
                _server_msg(
                    ServerMessageType.SECTION_TRANSITION,
                    from_section=previous_state.get("current_section_name"),
                    to_section=state.get("current_section_name"),
                )
            )

        await _send_section_info(outbound_queue, state)  # type: ignore

    bot_turns = _new_bot_turns(previous_state, state)

    if bot_turns:
        if len(bot_turns) == 1:
            turn = bot_turns[0]
            reply_text = str(turn.get("text") or "").strip()
            reply_type = str(turn.get("turn_type") or "assistant")
            turn_number = int(turn.get("turn_number") or state.get("turn_number") or 0)
        else:
            reply_text = " ".join(
                str(turn.get("text") or "").strip()
                for turn in bot_turns
                if turn.get("text")
            ).strip()
            reply_type = str(bot_turns[-1].get("turn_type") or "assistant")
            turn_number = int(
                bot_turns[-1].get("turn_number") or state.get("turn_number") or 0
            )

        if reply_text:
            await _send_bot_reply(
                outbound_queue,
                reply_text,
                reply_type,
                turn_number,
                bot_speaking_event,
            )

    if state.get("think_timer_active") and not (
        previous_state and previous_state.get("think_timer_active")
    ):
        await outbound_queue.put(
            _server_msg(ServerMessageType.THINK_TIMER_START, duration_secs=15)
        )

    if state.get("closing_done"):
        event_type = (
            ServerMessageType.SESSION_TERMINATED
            if state.get("session_status") == "TERMINATED"
            else ServerMessageType.INTERVIEW_COMPLETE
        )
        await outbound_queue.put(_server_msg(event_type))


@router.websocket("/ws/interview")
async def interview_websocket(websocket: WebSocket) -> None:
    candidate_assessment_id = (
        websocket.headers.get("x-candidate-assessment-id")
        or websocket.headers.get("x-assessment-id")
        or ""
    )

    if not candidate_assessment_id:
        await websocket.close(code=1008)
        return

    can_start, reason = await db.assert_session_can_start(candidate_assessment_id)

    if not can_start:
        await websocket.accept()
        await websocket.send_text(_server_msg(ServerMessageType.ERROR, message=reason))
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info("Interview websocket connected: ca_id=%s", candidate_assessment_id)

    orchestrator = InterviewOrchestrator(
        candidate_assessment_id=candidate_assessment_id
    )

    outbound_queue: asyncio.Queue[str | bytes | None] = asyncio.Queue()

    # Created immediately so early frontend audio chunks are not dropped.
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    session_state: dict[str, object] = {
        "started": False,
        "response_in_progress": False,
    }

    processing_lock = asyncio.Lock()

    # bot_speaking_event: SET = bot is silent (candidate's turn)
    #                     CLEARED = bot is speaking (candidate should wait)
    bot_speaking_event = asyncio.Event()
    bot_speaking_event.set()

    audio_pipeline_task: asyncio.Task | None = None

    await outbound_queue.put(
        _server_msg(
            ServerMessageType.CONNECTION_ACK,
            session_id=candidate_assessment_id,
            message="Interview session connected. Click Start Interview when you are ready.",
        )
    )

    def current_state() -> InterviewState | None:
        state = session_state.get("state")
        return state if isinstance(state, dict) else None  # type: ignore

    def cancel_watchdog() -> None:
        task = session_state.pop("watchdog_task", None)
        if isinstance(task, asyncio.Task) and not task.done():
            task.cancel()

    async def silence_watchdog(delay: float) -> None:
        try:
            await bot_speaking_event.wait()
            await asyncio.sleep(delay)

            state = current_state()

            if not state or _session_closed(state) or not session_state.get("started"):
                return

            await process_transcript("__SILENCE__")

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Silence watchdog failed")

    async def timeout_watchdog(total_budget_secs: int, total_elapsed: int) -> None:
        try:
            remaining = max(0, total_budget_secs - total_elapsed)

            if remaining > 0:
                await asyncio.sleep(remaining)

            state = current_state()

            if state and not _session_closed(state):
                await process_transcript("__TIME_UP__")

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Timeout watchdog failed")

    def schedule_watchdog(state: InterviewState) -> None:
        cancel_watchdog()

        if _session_closed(state) or state.get("next_node") != "await_response":
            return

        delay = 15.0 if state.get("think_timer_active") else 5.0

        session_state["watchdog_task"] = asyncio.create_task(
            silence_watchdog(delay),
            name="interview-silence-watchdog",
        )

    def schedule_timeout(state: InterviewState) -> None:
        if session_state.get("timeout_task"):
            return

        total_budget = int(state.get("total_interview_allocated_secs") or 0)

        if total_budget <= 0:
            return

        session_state["timeout_task"] = asyncio.create_task(
            timeout_watchdog(
                total_budget,
                int(state.get("total_elapsed_secs") or 0),
            ),
            name="interview-timeout-watchdog",
        )

    # ── Core transcript → graph submission ────────────────────────────────
    async def process_transcript(
        text: str,
        confidence: float | None = None,
    ) -> None:
        clean = (
            " ".join((text or "").split())
            if text not in {"__SILENCE__", "__TIME_UP__"}
            else text
        )

        if not clean or not session_state.get("started"):
            return

        if clean not in {"__SILENCE__", "__TIME_UP__"}:
            cancel_watchdog()

        # ── Wait for the bot to finish speaking before responding ─────
        # This is the core guard that prevents the bot from interrupting
        # itself.  The wait is non-blocking for the *inbound_task*
        # because process_transcript is only called from the audio
        # pipeline (background) or from the inline text_message handler.
        try:
            await asyncio.wait_for(bot_speaking_event.wait(), timeout=60.0)
        except TimeoutError:
            logger.warning("Timed out waiting for bot to finish speaking")
            bot_speaking_event.set()

        session_state["response_in_progress"] = True

        async with processing_lock:
            previous = current_state()

            if previous is None:
                session_state["response_in_progress"] = False
                return

            try:
                state = await orchestrator.submit_response(
                    clean,
                    stt_confidence=confidence,
                )

                session_state["state"] = state

                await _send_graph_outputs(
                    outbound_queue,
                    previous,
                    state,  # type: ignore
                    bot_speaking_event,
                )

                if not _session_closed(state):  # type: ignore
                    schedule_watchdog(state)  # type: ignore

            except Exception:
                logger.exception("Graph processing failed")
                await outbound_queue.put(
                    _server_msg(
                        ServerMessageType.ERROR,
                        message="AI response error. Please try again.",
                    )
                )
            finally:
                session_state["response_in_progress"] = False

    # ── Session lifecycle ─────────────────────────────────────────────────
    async def start_or_resume() -> None:
        async with processing_lock:
            previous = current_state()

            state = await orchestrator.resume_session()

            if state is not None:
                await outbound_queue.put(
                    _server_msg(
                        ServerMessageType.SESSION_RESUMED,
                        turn_number=state.get("turn_number", 0),
                        section_index=state.get("current_section_index", 0),
                        section_name=state.get("current_section_name", "unknown"),
                        message="Welcome back. Your interview session has been restored.",
                    )
                )
            else:
                state = await orchestrator.start_session()

            session_state["state"] = state
            session_state["started"] = True

            await _send_graph_outputs(
                outbound_queue,
                previous,
                state,  # type: ignore
                bot_speaking_event,
            )

            schedule_watchdog(state)  # type: ignore
            schedule_timeout(state)  # type: ignore

    async def close_session() -> None:
        cancel_watchdog()

        async with processing_lock:
            previous = current_state()

            if previous is None or previous.get("closing_done"):
                return

            state = await orchestrator.close_session()

            if state is not None:
                session_state["state"] = state
                await _send_graph_outputs(
                    outbound_queue,
                    previous,
                    state,  # type: ignore
                    bot_speaking_event,
                )

    # ── Audio pipeline (STT) ──────────────────────────────────────────────
    async def audio_pipeline() -> None:
        async def drain_audio(stt: STTService) -> None:
            """Read audio chunks from the queue and feed them to Deepgram.
            This must NEVER be blocked by bot_speaking_event — Deepgram
            needs continuous audio to keep the connection alive."""
            while True:
                chunk = await audio_queue.get()

                if chunk is None:
                    break

                await stt.send_audio(chunk)

        async def process_transcripts(stt: STTService) -> None:
            accumulated_text: list[str] = []
            timer_task: asyncio.Task | None = None
            latest_confidence: float | None = None

            def append_segment(segment: str) -> None:
                clean = " ".join(segment.split())

                if not clean:
                    return

                current = " ".join(accumulated_text).strip()

                if clean == current or current.endswith(clean):
                    return

                if current and clean.startswith(current):
                    accumulated_text.clear()
                    accumulated_text.append(clean)
                    return

                accumulated_text.append(clean)

            # ── Minimum commit delay (seconds) ────────────────────────
            # All candidate turns wait at least this long after the last
            # final STT segment before being committed to the graph.
            COMMIT_DELAY_SECS = 1.5

            async def wait_and_trigger(delay: float = COMMIT_DELAY_SECS) -> None:
                nonlocal latest_confidence

                try:
                    if delay > 0:
                        await asyncio.sleep(delay)

                    # ── Wait for bot to finish speaking ───────────────
                    # Instead of skipping the commit entirely (which
                    # would lose the accumulated text), we *wait* for the
                    # bot to finish, then commit.
                    try:
                        await asyncio.wait_for(bot_speaking_event.wait(), timeout=60.0)
                    except TimeoutError:
                        logger.warning("Commit: timed out waiting for bot to finish")
                        bot_speaking_event.set()

                    # Don't double-submit while another response is in flight
                    if session_state.get("response_in_progress"):
                        logger.debug("Skipping commit — response already in progress")
                        return

                    combined_text = " ".join(accumulated_text).strip()
                    accumulated_text.clear()

                    if not combined_text:
                        return

                    await outbound_queue.put(
                        _server_msg(
                            ServerMessageType.FINAL_TRANSCRIPT,
                            text=combined_text,
                        )
                    )

                    confidence = latest_confidence
                    latest_confidence = None

                    # Fire and forget so that if wait_and_trigger gets cancelled by a new STT event,
                    # we don't cancel the running LangGraph execution.
                    asyncio.create_task(
                        process_transcript(
                            combined_text,
                            confidence=confidence,
                        ),
                        name="interview-process-transcript",
                    )

                except asyncio.CancelledError:
                    pass

            try:
                async for event in stt.transcript_events():
                    text = " ".join((event.text or "").split())

                    # ── ALWAYS accumulate STT events ──────────────────
                    # We never drop events.  Deepgram must always be fed
                    # and its output must always be captured.  The
                    # bot_speaking_event gate is only in wait_and_trigger
                    # (commit) and process_transcript (graph submission).

                    latest_confidence = getattr(
                        event,
                        "confidence",
                        latest_confidence,
                    )

                    # Only cancel the silence watchdog if the bot is NOT
                    # speaking (otherwise the bot's own audio echoed back
                    # through the mic would cancel it).
                    if text and bot_speaking_event.is_set():
                        cancel_watchdog()

                    if timer_task and not timer_task.done():
                        timer_task.cancel()

                    if not event.is_final:
                        display_text = " ".join([*accumulated_text, text]).strip()

                        if display_text and bot_speaking_event.is_set():
                            await outbound_queue.put(
                                _server_msg(
                                    ServerMessageType.PARTIAL_TRANSCRIPT,
                                    text=display_text,
                                )
                            )

                        continue

                    if text:
                        append_segment(text)

                    display_text = " ".join(accumulated_text).strip()

                    if display_text and bot_speaking_event.is_set():
                        await outbound_queue.put(
                            _server_msg(
                                ServerMessageType.PARTIAL_TRANSCRIPT,
                                text=display_text,
                            )
                        )

                    # Only schedule the commit timer if there is text.
                    # We always schedule it. If the bot is speaking, wait_and_trigger
                    # will pause at `await bot_speaking_event.wait()` and commit
                    # after the bot finishes.
                    if accumulated_text:
                        timer_task = asyncio.create_task(
                            wait_and_trigger(COMMIT_DELAY_SECS)
                        )

            finally:
                if timer_task and not timer_task.done():
                    timer_task.cancel()

        def _detect_encoding(chunk: bytes) -> tuple[str | None, str | int | None]:
            """Sniff the first audio chunk to pick Deepgram stream config."""
            if chunk[:4] in (b"\x1a\x45\xdf\xa3", b"OggS", b"RIFF"):
                logger.info(
                    "Detected container audio (WebM/Ogg/WAV); using Deepgram auto-detect"
                )
                return None, None
            logger.info("Assuming raw linear16 PCM browser audio")
            return "linear16", 16000

        # Wait for the first audio chunk before connecting to Deepgram.
        # This avoids the "no audio within timeout" error that occurs when
        # the STT connection is opened before the browser starts streaming.
        logger.info("Audio pipeline ready; waiting for first audio chunk")
        first_chunk = await audio_queue.get()
        if first_chunk is None:
            return

        encoding, sample_rate = _detect_encoding(first_chunk)

        async with STTService(encoding=encoding, sample_rate=sample_rate) as stt:
            # Send the first chunk that we already consumed
            await stt.send_audio(first_chunk)
            logger.info(
                "First audio chunk sent to Deepgram: bytes=%d encoding=%s",
                len(first_chunk),
                encoding or "auto",
            )
            await asyncio.gather(
                drain_audio(stt),
                process_transcripts(stt),
            )

    # ── Inbound task (reads WebSocket frames) ─────────────────────────────
    async def inbound_task() -> None:
        nonlocal audio_pipeline_task

        while True:
            try:
                frame = await websocket.receive()
            except WebSocketDisconnect:
                break

            if frame.get("type") == "websocket.disconnect":
                break

            # Audio chunks — always forward to audio_queue immediately
            if frame.get("bytes"):
                await audio_queue.put(frame["bytes"])
                continue

            raw_text = frame.get("text")

            if not raw_text:
                continue

            try:
                msg = ClientMessage.model_validate(json.loads(raw_text))
            except Exception:
                await outbound_queue.put(
                    _server_msg(
                        ServerMessageType.ERROR,
                        message="Invalid message format.",
                    )
                )
                continue

            if msg.type == ClientMessageType.PING:
                await outbound_queue.put(_server_msg(ServerMessageType.PONG))

            elif msg.type in {
                ClientMessageType.SESSION_START,
                ClientMessageType.SESSION_RESUME,
            }:
                if audio_pipeline_task is None or audio_pipeline_task.done():
                    audio_pipeline_task = asyncio.create_task(
                        audio_pipeline(),
                        name="interview-audio-pipeline",
                    )

                # ── CRITICAL: launch as background task ───────────────
                # start_or_resume calls _send_bot_reply which clears
                # bot_speaking_event.  We must NOT await it inline
                # because inbound_task must keep reading WebSocket
                # frames (especially audio) without stalling.
                if not session_state.get("started"):
                    asyncio.create_task(
                        _safe_start_or_resume(),
                        name="interview-start-or-resume",
                    )

            elif msg.type == ClientMessageType.TEXT_MESSAGE:
                text = str(msg.payload.get("text") or "").strip()

                if text:
                    await outbound_queue.put(
                        _server_msg(
                            ServerMessageType.FINAL_TRANSCRIPT,
                            text=text,
                        )
                    )
                    # Also launch as background so inbound loop stays free
                    asyncio.create_task(
                        process_transcript(text),
                        name="interview-text-response",
                    )

            elif msg.type == ClientMessageType.STOP:
                await close_session()
                break

        await outbound_queue.put(None)
        await audio_queue.put(None)

        cancel_watchdog()

        timeout = session_state.get("timeout_task")

        if isinstance(timeout, asyncio.Task) and not timeout.done():
            timeout.cancel()

    async def _safe_start_or_resume() -> None:
        """Wrapper so start_or_resume exceptions don't crash the event loop."""
        try:
            await start_or_resume()
        except Exception:
            logger.exception("Failed to start/resume interview session")
            await outbound_queue.put(
                _server_msg(
                    ServerMessageType.ERROR,
                    message="Failed to initialise interview. Please try again.",
                )
            )

    # ── Outbound task (sends WebSocket frames) ────────────────────────────
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
                logger.exception("Error sending interview websocket frame")
                break

    tasks = {
        asyncio.create_task(inbound_task(), name="interview-inbound"),
        asyncio.create_task(outbound_task(), name="interview-outbound"),
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
            logger.exception("Unhandled interview websocket task error")

    try:
        state = current_state()

        if state and not state.get("closing_done"):
            await orchestrator.pause_session()

    except Exception:
        logger.exception("Failed to pause interview session on disconnect")

    logger.info("Interview websocket disconnected: ca_id=%s", candidate_assessment_id)
