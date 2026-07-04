"""LiveKit voice agent worker for AI interviews.

Deepgram Nova-3 provides transcript quality, while LiveKit TurnDetector owns
end-of-turn detection before LangGraph receives a candidate response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import AsyncIterable
from typing import Any, cast

from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ChatContext,
    ChatMessage,
    ModelSettings,
    StopResponse,
    TurnHandlingOptions,
    inference,
    llm,
)
from livekit.plugins import deepgram

from src.config.settings import settings
from src.control.agents.nodes.classify_response import will_bypass_interviewer_llm
from src.control.agents.nodes.question_strategy import is_self_intro_phase
from src.control.agents.state import InterviewState
from src.control.agents.templates import choose_template_avoiding
from src.core.services.livekit_graph_bridge import LiveKitInterviewBridge
from src.utils.livekit import chunk_for_tts

logger = logging.getLogger("interview-livekit-agent")
USER_AWAY_TIMEOUT_SECS = 5.0
THINK_EXTENSION_TIMEOUT_SECS = 15.0
POST_TURN_SILENCE_GUARD_SECS = 0.5
POST_BARGE_RESUMED_SPEECH_GUARD_SECS = 0.75
DEMO_GREETING = (
    "Welcome to this demo interview. This is a short practice space to help "
    "you become comfortable with the interview environment. Please make "
    "yourself comfortable, and say anything when you are ready."
)
DEMO_RESPONSE_TEMPLATES = (
    "Thank you. Your response came through clearly, and you can continue whenever you are ready.",
    "Great, I heard you clearly. Feel free to say a little more so you can get used to the experience.",
    "That came through well. This practice room works just like the live voice interview.",
    "Thank you for sharing that. Take your time and continue speaking whenever you feel comfortable.",
    "Perfect, your microphone and the interview connection are working as expected.",
    "I heard your response clearly. You can keep practicing at your own pace.",
    "Thanks, that sounded clear. This is a good opportunity to become familiar with the response flow.",
    "Your response was received successfully. Feel free to try another answer when you are ready.",
    "Everything is coming through properly. You can continue speaking naturally, just as you would in the interview.",
    "Thank you. The practice setup is working well, and you may continue whenever you would like.",
)


def prewarm(proc: agents.JobProcess) -> None:
    """
    Load expensive AI models into memory once per worker process.
    This ensures models like Silero VAD are ready before any interviews start.

    Args:
        proc: The JobProcess instance from LiveKit.
    """
    proc.userdata["vad"] = inference.VAD(
        model="silero",
        min_speech_duration=settings.VAD_MIN_SPEECH_DURATION_SECS,
        min_silence_duration=settings.VAD_MIN_SILENCE_DURATION_SECS,
        prefix_padding_duration=settings.VAD_PREFIX_PADDING_DURATION_SECS,
    )


server = AgentServer(
    ws_url=settings.LIVEKIT_URL or None,
    api_key=settings.LIVEKIT_API_KEY or None,
    api_secret=settings.LIVEKIT_API_SECRET or None,
    setup_fnc=prewarm,
)


async def _stream_text_chunks(
    text: str,
    *,
    max_chars: int = 96,
) -> AsyncIterable[str]:
    """Yield punctuation-aware chunks to the TTS pipeline."""

    for chunk in chunk_for_tts(text, max_chars=max_chars):
        yield chunk


class InternalPipelineLLM(llm.LLM):  # type: ignore[misc]
    """Required LiveKit adapter for agents with custom response nodes."""

    def __init__(self, model_name: str) -> None:
        """Initialize the non-provider adapter with a diagnostic model name."""

        super().__init__()
        self._model_name = model_name

    @property
    def model(self) -> str:
        """Return the diagnostic model name."""

        return self._model_name

    @property
    def provider(self) -> str:
        """Return the internal provider label."""

        return "internal"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: Any = agents.DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: Any = agents.NOT_GIVEN,
        tool_choice: Any = agents.NOT_GIVEN,
        extra_kwargs: Any = agents.NOT_GIVEN,
    ) -> llm.LLMStream:
        """
        Reject the fallback provider path.

        Raises:
            RuntimeError: Always, because the agent implements ``llm_node``.
        """
        del (
            chat_ctx,
            tools,
            conn_options,
            parallel_tool_calls,
            tool_choice,
            extra_kwargs,
        )
        raise RuntimeError("This agent only supports its custom response node")


class DemoLiveKitAgent(Agent):  # type: ignore[misc]
    """Disposable practice agent driven only by static response templates."""

    def __init__(self) -> None:
        """Initialize static demo response state."""

        super().__init__(
            instructions="Static no-LLM demo interview agent.",
            llm=InternalPipelineLLM("static-demo-templates"),
        )
        self._response_pending = False
        self._last_response_index: int | None = None

    async def on_enter(self) -> None:
        """
        Speak the fixed welcome message when the LiveKit session connects.
        This allows the user to know the demo room is active.
        """

        handle = self.session.say(
            _stream_text_chunks(DEMO_GREETING),
            allow_interruptions=True,
            add_to_chat_ctx=True,
        )
        await handle.wait_for_playout()

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        """
        Accept any non-empty speech from the candidate. This signals that the demo
        agent needs to formulate a response.

        Args:
            turn_ctx: The context of the ongoing chat.
            new_message: The finalized message transcribed from the user's speech.

        Raises:
            StopResponse: If the text is empty.
        """
        del turn_ctx

        candidate_text = " ".join((new_message.text_content or "").split())
        if not candidate_text:
            raise StopResponse()
        self._response_pending = True

    def _choose_response(self) -> str:
        """
        Select a random demo response template, avoiding repeating the previous one.

        Returns:
            The selected response text.
        """
        available_indices = [
            index
            for index in range(len(DEMO_RESPONSE_TEMPLATES))
            if index != self._last_response_index
        ]
        selected_index = random.choice(available_indices)
        self._last_response_index = selected_index
        return DEMO_RESPONSE_TEMPLATES[selected_index]

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str]:
        """
        Provide the response for the demo agent. This yields a static phrase
        instead of calling out to a real LLM.

        Args:
            chat_ctx: The ongoing chat context.
            tools: Tools available (unused here).
            model_settings: Model configuration (unused here).

        Yields:
            String chunks for the TTS engine.
        """
        del chat_ctx, tools, model_settings

        if not self._response_pending:
            return
        self._response_pending = False
        for chunk in chunk_for_tts(self._choose_response()):
            yield chunk


class InterviewLiveKitAgent(Agent):  # type: ignore[misc]
    """LiveKit voice agent that delegates interview reasoning to LangGraph."""

    def __init__(
        self,
        *,
        candidate_assessment_id: str,
        interview_session_id: str,
        connection_id: str,
    ) -> None:
        """Initialize graph, timing, transcript, and playout state."""

        super().__init__(
            instructions="Conduct the interview through the LangGraph workflow.",
            llm=InternalPipelineLLM("langgraph-interview"),
        )

        self.bridge = LiveKitInterviewBridge(
            candidate_assessment_id=candidate_assessment_id,
            interview_session_id=interview_session_id,
            connection_id=connection_id,
        )

        self._pending_user_text: str | None = None
        self._pending_duration_ms: int | None = None
        self._user_turn_started_at: float | None = None
        self._last_user_turn_duration_ms: int | None = None
        self._last_user_turn_finished_at: float | None = None
        self._interview_closed = False
        self._agent_is_speaking = False
        self._last_agent_speech_finished_at: float | None = None
        self._pending_silence_task: asyncio.Task[None] | None = None
        self._awaiting_agent_reply = False
        self._timer_start_requested = False
        self._pending_section_barge_task: asyncio.Task[None] | None = None
        self._barge_in_active = False
        self._discard_inflight_user_turn = False
        self._post_barge_guard_deadline: float | None = None
        self._live_user_transcript = ""
        self._committed_user_transcript = ""
        self._current_user_interim = ""
        self._closing_finalize_requested = False
        self._turn_lock = asyncio.Lock()
        self._recent_fillers: list[str] = []

    def _mark_closed_from_state(self) -> bool:
        """
        Check the interview state from the bridge to determine if the interview is closed.

        Returns:
            True if the interview has completed or is closing, False otherwise.
        """
        state = self.bridge.state or {}
        self._interview_closed = bool(
            state.get("closing_done")
            or state.get("should_close")
            or state.get("bot_reply_type") == "closing"
        )
        return self._interview_closed

    async def _finalize_after_closing(self) -> None:
        """
        Finalize the interview process once the closing audio has finished playing.
        This disables audio input and triggers the bridge to save final data.
        """
        if self._closing_finalize_requested:
            return
        self._closing_finalize_requested = True
        self._cancel_pending_section_barge()
        try:
            self.session.input.set_audio_enabled(False)
            await self.bridge.finalize_closing()
        except Exception:
            logger.exception(
                "Failed to finalize interview after closing",
                extra={"candidate_assessment_id": self.bridge.candidate_assessment_id},
            )
        try:
            self.session.shutdown(drain=True)
        except Exception:
            logger.exception(
                "Failed to shutdown LiveKit session after closing",
                extra={"candidate_assessment_id": self.bridge.candidate_assessment_id},
            )

    def _cancel_pending_silence(self) -> None:
        task = self._pending_silence_task
        current_task = asyncio.current_task()
        if task and task is not current_task and not task.done():
            task.cancel()
        self._pending_silence_task = None

    def _mark_agent_speech_finished(self) -> None:
        self._agent_is_speaking = False
        self._last_agent_speech_finished_at = time.monotonic()

    async def _start_timer_after_first_speech(self) -> None:
        try:
            await self.bridge.start_timer()
            self._schedule_section_barge_watchdog()
        except Exception:
            logger.exception(
                "Failed to start interview timer on first bot speech",
                extra={"candidate_assessment_id": self.bridge.candidate_assessment_id},
            )

    def _cancel_pending_section_barge(self) -> None:
        task = self._pending_section_barge_task
        current_task = asyncio.current_task()
        if task and task is not current_task and not task.done():
            task.cancel()
        self._pending_section_barge_task = None

    async def _delayed_section_barge(self, delay_secs: float) -> None:
        try:
            await asyncio.sleep(max(0.0, delay_secs))
            await self.handle_section_time_barge_in()
        except asyncio.CancelledError:
            return

    def _schedule_section_barge_watchdog(self) -> None:
        if (
            self._interview_closed
            or self._barge_in_active
            or self._agent_is_speaking
            or not self._timer_start_requested
        ):
            return
        delay = self.bridge.seconds_until_section_barge_in()
        if delay is None:
            return
        self._cancel_pending_section_barge()
        self._pending_section_barge_task = asyncio.create_task(
            self._delayed_section_barge(delay)
        )

    def _seconds_until_silence_allowed(self) -> float:
        state = self.bridge.state or {}
        timeout_secs = (
            THINK_EXTENSION_TIMEOUT_SECS
            if state.get("bot_reply_type") == "think_wait"
            else USER_AWAY_TIMEOUT_SECS
        )
        if self._agent_is_speaking or self._last_agent_speech_finished_at is None:
            return timeout_secs
        elapsed = time.monotonic() - self._last_agent_speech_finished_at
        return max(0.0, timeout_secs - elapsed)

    async def _delayed_silence_check(self, delay_secs: float) -> None:
        try:
            await asyncio.sleep(max(0.0, delay_secs))
            await self.handle_user_away()
        except asyncio.CancelledError:
            return

    def _schedule_delayed_silence(self, delay_secs: float) -> None:
        task = self._pending_silence_task
        if task and not task.done():
            return
        self._pending_silence_task = asyncio.create_task(
            self._delayed_silence_check(delay_secs)
        )

    async def _say_text(
        self,
        text: str,
        *,
        allow_interruptions: bool,
        add_to_chat_ctx: bool = True,
    ) -> None:
        self._cancel_pending_silence()
        self._agent_is_speaking = True
        handle = self.session.say(
            _stream_text_chunks(text),
            allow_interruptions=allow_interruptions,
            add_to_chat_ctx=add_to_chat_ctx,
        )
        await handle.wait_for_playout()
        self._mark_agent_speech_finished()

    async def _say_graph_reply(
        self,
        text: str,
        *,
        is_closing: bool,
        allow_interruptions: bool | None = None,
    ) -> None:
        """Speak a graph reply and finalize the session after a closing."""

        if not text:
            return
        await self._say_text(
            text,
            allow_interruptions=(
                not is_closing if allow_interruptions is None else allow_interruptions
            ),
            add_to_chat_ctx=True,
        )
        if is_closing:
            await self._finalize_after_closing()

    def track_agent_state(self, new_state: str) -> None:
        """
        Track agent speech so user-away silence and timers start after bot audio ends.

        Args:
            new_state: The new state of the agent ('speaking', 'listening', etc.).
        """

        if new_state == "speaking":
            self._cancel_pending_silence()
            if not self._barge_in_active:
                self._cancel_pending_section_barge()
            self._agent_is_speaking = True
            self._awaiting_agent_reply = False
            if not self._timer_start_requested:
                state = self.bridge.state or {}
                is_opening_turn = int(state.get("turn_number") or 1) <= 1 and str(
                    state.get("bot_reply_type") or "opening"
                ) in {"opening", ""}
                if is_opening_turn:
                    self._timer_start_requested = True
                    asyncio.create_task(self._start_timer_after_first_speech())
            return

        if self._agent_is_speaking:
            self._mark_agent_speech_finished()
            if not self._interview_closed and not self._awaiting_agent_reply:
                self._schedule_delayed_silence(USER_AWAY_TIMEOUT_SECS)
            if self._interview_closed:
                if not self._closing_finalize_requested:
                    asyncio.create_task(self._finalize_after_closing())
            elif not self._barge_in_active:
                self._schedule_section_barge_watchdog()

    async def on_enter(self) -> None:
        """
        Start or resume the interview and speak the opening graph turn.
        This executes when the LiveKit session is fully established.
        """

        try:
            opening_text = await self.bridge.start_or_resume()
        except Exception:
            logger.exception(
                "Failed to start or resume interview graph",
                extra={"candidate_assessment_id": self.bridge.candidate_assessment_id},
            )
            opening_text = (
                "I am connected now, but I had a brief issue preparing the interview. "
                "Please wait a moment while I recover."
            )

        self._mark_closed_from_state()

        if not opening_text:
            logger.warning(
                "Interview graph returned no opening text",
                extra={"candidate_assessment_id": self.bridge.candidate_assessment_id},
            )
            return

        await self._say_graph_reply(
            opening_text,
            is_closing=self._interview_closed,
        )

    def track_user_state(self, new_state: str) -> None:
        """
        Record user speech duration for optional graph/timing metadata and handle
        the transition out of silence.

        Args:
            new_state: The new state of the user ('speaking', 'listening', etc.).
        """

        if new_state == "speaking":
            self._cancel_pending_silence()
            now = time.monotonic()
            if (
                self._post_barge_guard_deadline is not None
                and now <= self._post_barge_guard_deadline
            ):
                # Disabling audio during a timed section transition forces
                # LiveKit into "listening". If the interrupted candidate is
                # still talking when audio resumes, LiveKit reports it as a
                # fresh turn. Discard that resumed fragment instead of
                # classifying it against the next section's question.
                self._discard_inflight_user_turn = True
                self._post_barge_guard_deadline = None
                logger.info(
                    "Discarding speech resumed across section transition",
                    extra={
                        "candidate_assessment_id": (self.bridge.candidate_assessment_id)
                    },
                )
            elif self._post_barge_guard_deadline is not None:
                self._post_barge_guard_deadline = None
                self._discard_inflight_user_turn = False
            elif self._discard_inflight_user_turn and not self._barge_in_active:
                # A new post-transition turn is valid; only the interrupted turn
                # should be discarded.
                self._discard_inflight_user_turn = False

        if new_state == "speaking" and self._user_turn_started_at is None:
            self._user_turn_started_at = time.monotonic()
            self._last_user_turn_finished_at = None
            return

        if new_state == "speaking" or self._user_turn_started_at is None:
            return

        self._last_user_turn_duration_ms = int(
            (time.monotonic() - self._user_turn_started_at) * 1000
        )
        self._user_turn_started_at = None
        self._last_user_turn_finished_at = time.monotonic()

    def track_user_transcription(self, text: str, *, is_final: bool) -> None:
        """
        Accumulate final STT chunks and retain the latest interim suffix.

        Args:
            text: The text transcribed so far in the chunk.
            is_final: Whether this chunk represents a completed, finalized phrase.
        """

        normalized = " ".join((text or "").split())
        if is_final:
            if normalized:
                committed = self._committed_user_transcript
                if not committed or normalized.casefold().startswith(
                    committed.casefold()
                ):
                    self._committed_user_transcript = normalized
                elif not committed.casefold().endswith(normalized.casefold()):
                    self._committed_user_transcript = (
                        f"{committed} {normalized}".strip()
                    )
            self._current_user_interim = ""
        else:
            self._current_user_interim = normalized

        committed = self._committed_user_transcript
        interim = self._current_user_interim
        if (
            committed
            and interim
            and interim.casefold().startswith(committed.casefold())
        ):
            self._live_user_transcript = interim
        else:
            self._live_user_transcript = " ".join(
                item for item in (committed, interim) if item
            ).strip()

    def _clear_live_user_transcript(self) -> None:
        self._live_user_transcript = ""
        self._committed_user_transcript = ""
        self._current_user_interim = ""

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        """
        Capture only LiveKit-confirmed user turns for graph submission.

        Args:
            turn_ctx: The context of the ongoing chat.
            new_message: The finalized message transcribed from the user's speech.

        Raises:
            StopResponse: If the turn is discarded or the interview is closed.
        """
        del turn_ctx

        if self._interview_closed:
            raise StopResponse()

        if self._barge_in_active or self._discard_inflight_user_turn:
            self._discard_inflight_user_turn = False
            self._user_turn_started_at = None
            self._last_user_turn_duration_ms = None
            self._last_user_turn_finished_at = time.monotonic()
            self._clear_live_user_transcript()
            raise StopResponse()

        candidate_text = " ".join((new_message.text_content or "").split())

        if not candidate_text:
            raise StopResponse()

        if self._user_turn_started_at is not None:
            self.track_user_state("listening")

        self._pending_duration_ms = self._last_user_turn_duration_ms
        self._last_user_turn_duration_ms = None
        self._pending_user_text = candidate_text
        self._clear_live_user_transcript()
        self._awaiting_agent_reply = True
        self._cancel_pending_silence()

    async def handle_user_away(self) -> None:
        """
        Recover buffered speech, or submit silence when no speech exists, to move
        the graph forward if the user has been silent for too long.
        """

        if self._interview_closed:
            return

        state = self.bridge.state or {}
        if state.get("phase_complete") or state.get("should_advance_question"):
            # Do not start a silence flow while transitioning or closing.
            return

        if (
            self._pending_user_text
            or self._awaiting_agent_reply
            or self._turn_lock.locked()
        ):
            return

        buffered_candidate_text = " ".join(self._live_user_transcript.split())
        if buffered_candidate_text and is_self_intro_phase(state):
            logger.info(
                "Skipping buffered speech recovery during self-introduction",
                extra={
                    "candidate_assessment_id": self.bridge.candidate_assessment_id,
                    "text_length": len(buffered_candidate_text),
                },
            )
            return

        if buffered_candidate_text:
            self._cancel_pending_silence()
            async with self._turn_lock:
                if self._pending_user_text or self._awaiting_agent_reply:
                    return
                buffered_candidate_text = " ".join(self._live_user_transcript.split())
                if not buffered_candidate_text:
                    return

                # LiveKit occasionally emits complete STT chunks without
                # finalizing a turn. Recover that speech as an answer and
                # suppress a later duplicate turn-completed callback.
                self._discard_inflight_user_turn = True
                self._awaiting_agent_reply = True
                duration_ms = self._last_user_turn_duration_ms
                self._last_user_turn_duration_ms = None
                self._clear_live_user_transcript()
                logger.info(
                    "Recovering buffered candidate speech before silence handling",
                    extra={
                        "candidate_assessment_id": (
                            self.bridge.candidate_assessment_id
                        ),
                        "text_length": len(buffered_candidate_text),
                    },
                )
                try:
                    reply_text = await self.bridge.submit_candidate_turn(
                        buffered_candidate_text,
                        duration_ms=duration_ms,
                    )
                    is_closing_reply = self._mark_closed_from_state()
                except Exception:
                    logger.exception(
                        "Buffered candidate-turn recovery failed",
                        extra={
                            "candidate_assessment_id": (
                                self.bridge.candidate_assessment_id
                            )
                        },
                    )
                    reply_text = (
                        "I had a brief issue processing that response. "
                        "Let's continue with the next question."
                    )
                    is_closing_reply = False
                finally:
                    self._awaiting_agent_reply = False

            await self._say_graph_reply(
                reply_text,
                is_closing=is_closing_reply,
            )
            return

        if self._last_user_turn_finished_at is not None:
            turn_completion_wait = POST_TURN_SILENCE_GUARD_SECS - (
                time.monotonic() - self._last_user_turn_finished_at
            )
            if turn_completion_wait > 0.05:
                self._schedule_delayed_silence(turn_completion_wait)
                return

        remaining_silence_wait = self._seconds_until_silence_allowed()
        if remaining_silence_wait > 0.05:
            self._schedule_delayed_silence(remaining_silence_wait)
            return

        async with self._turn_lock:
            reply_text = await self.bridge.submit_silence()
            is_closing_reply = self._mark_closed_from_state()

        await self._say_graph_reply(
            reply_text,
            is_closing=is_closing_reply,
        )

    async def handle_session_close(self, reason: str) -> None:
        """
        Persist unexpected room closure without affecting LiveKit teardown.

        Args:
            reason: The reason for closure provided by LiveKit.
        """

        try:
            await self.bridge.record_disconnect(reason)
        except Exception:
            logger.exception(
                "Failed to record LiveKit session disconnection",
                extra={
                    "candidate_assessment_id": (self.bridge.candidate_assessment_id)
                },
            )

    async def handle_section_time_barge_in(self) -> None:
        """
        Interrupt an overrun section, including during candidate speech, to force
        the interview to progress to the next phase on schedule.
        """

        if self._interview_closed or self._barge_in_active:
            return

        had_inflight_user_turn = bool(
            self._user_turn_started_at is not None or self._live_user_transcript
        )
        remaining = self.bridge.seconds_until_section_barge_in()
        if remaining is None:
            return
        if remaining > 0.1:
            self._schedule_section_barge_watchdog()
            return

        self._barge_in_active = True
        self._discard_inflight_user_turn = had_inflight_user_turn
        self._cancel_pending_silence()
        try:
            try:
                self.session.input.set_audio_enabled(False)
            except RuntimeError:
                # Agent is no longer running, abort barge-in
                return

            async with self._turn_lock:
                remaining = self.bridge.seconds_until_section_barge_in()
                if remaining is None or remaining > 0.1:
                    return
                reply_text = await self.bridge.submit_section_time_barge_in(
                    partial_candidate_text=self._live_user_transcript,
                )
                self._clear_live_user_transcript()
                is_closing_reply = self._mark_closed_from_state()

            await self._say_graph_reply(
                reply_text,
                is_closing=is_closing_reply,
                allow_interruptions=False,
            )
        except Exception:
            logger.exception(
                "Section time barge-in failed",
                extra={"candidate_assessment_id": self.bridge.candidate_assessment_id},
            )
        finally:
            self._barge_in_active = False
            if not self._interview_closed:
                self.session.input.set_audio_enabled(True)
                if had_inflight_user_turn:
                    self._post_barge_guard_deadline = (
                        time.monotonic() + POST_BARGE_RESUMED_SPEECH_GUARD_SECS
                    )
                self._schedule_section_barge_watchdog()

    def _choose_filler(self) -> str:
        """
        Pick a short, content-neutral acknowledgement that is safe before any kind
        of candidate response, avoiding the few most recently spoken fillers.

        Returns:
            The chosen filler phrase.
        """

        filler = choose_template_avoiding(
            "neutral_filler",
            recent=self._recent_fillers,
        )
        self._recent_fillers = [*self._recent_fillers, filler][-4:]
        return filler

    async def _run_candidate_turn(
        self,
        candidate_text: str,
        duration_ms: int | None,
    ) -> tuple[str, bool]:
        """
        Resume the LangGraph workflow for one candidate turn under the turn lock.

        Args:
            candidate_text: The candidate's transcribed speech.
            duration_ms: How long the candidate spoke, if known.

        Returns:
            A tuple of (reply_text, is_closing_reply).
        """

        async with self._turn_lock:
            try:
                reply_text = await self.bridge.submit_candidate_turn(
                    candidate_text,
                    duration_ms=duration_ms,
                )
                is_closing_reply = self._mark_closed_from_state()
            except Exception:
                logger.exception("LangGraph turn failed")
                reply_text = (
                    "I had a brief issue processing that response. "
                    "Let's continue with the next question."
                )
                is_closing_reply = False
        return reply_text, is_closing_reply

    def _extract_candidate_text(self, chat_ctx: llm.ChatContext) -> str:
        """Resolve the latest user transcript from chat context or live STT buffer."""

        items = getattr(chat_ctx, "items", None) or []
        for item in reversed(items):
            if getattr(item, "role", None) == "user":
                text = getattr(item, "text_content", None) or ""
                cleaned = " ".join(str(text).split())
                if cleaned:
                    return cleaned
        return " ".join(self._live_user_transcript.split())

    async def _warm_speculative_turn(self, candidate_text: str) -> None:
        try:
            await self.bridge.warm_speculative_turn(candidate_text)
        except Exception:
            logger.exception(
                "Speculative interviewer warmup failed",
                extra={"candidate_assessment_id": self.bridge.candidate_assessment_id},
            )

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str]:
        """
        Generate assistant text by resuming the existing LangGraph workflow with
        the candidate's transcribed speech.

        A short, content-neutral acknowledgement is spoken immediately while the
        single merged interviewer-turn call runs concurrently, so the candidate
        hears a human-like reply with no perceptible dead air before the question.

        Args:
            chat_ctx: The chat context for preemptive or final user text.
            tools: Tools available (unused).
            model_settings: Model settings (unused).

        Yields:
            Text chunks for the TTS engine.
        """
        del tools, model_settings

        candidate_text = self._pending_user_text
        duration_ms = self._pending_duration_ms
        is_final_turn = candidate_text is not None

        if is_final_turn:
            self._pending_user_text = None
            self._pending_duration_ms = None
        else:
            candidate_text = self._extract_candidate_text(chat_ctx)
            duration_ms = self._last_user_turn_duration_ms

        if self._interview_closed:
            return

        if not candidate_text:
            return

        if not is_final_turn:
            preview = {
                **(self.bridge.state or {}),
                "previous_candidate_response": candidate_text,
            }
            if settings.PREEMPTIVE_GENERATION_ENABLED and not is_self_intro_phase(
                preview
            ):
                asyncio.create_task(self._warm_speculative_turn(candidate_text))
            return

        try:
            # Start the reasoning call first so it overlaps with the filler playout.
            graph_task = asyncio.create_task(
                self._run_candidate_turn(candidate_text, duration_ms)
            )

            preview_state = cast(
                InterviewState,
                {
                    **(self.bridge.state or {}),
                    "previous_candidate_response": candidate_text,
                },
            )
            skip_filler = will_bypass_interviewer_llm(preview_state, candidate_text)
            if settings.ENABLE_ACK_FILLER and not skip_filler:
                for chunk in chunk_for_tts(self._choose_filler()):
                    yield chunk

            reply_text, is_closing_reply = await graph_task

            if not reply_text:
                return

            if is_closing_reply:
                # Disable microphone ingestion before the first closing audio
                # frame so the final statement cannot be interrupted.
                self.session.input.set_audio_enabled(False)

            for chunk in chunk_for_tts(reply_text):
                yield chunk
        finally:
            self._awaiting_agent_reply = False


async def request_fnc(req: agents.JobRequest) -> None:
    """
    Accept explicitly dispatched interview-agent jobs.

    Args:
        req: The incoming job request from LiveKit.
    """

    await req.accept(
        name="AI Interview Bot",
        identity=f"interview-bot-{req.id}",
    )


@server.rtc_session(
    agent_name=settings.LIVEKIT_AGENT_NAME,
    on_request=request_fnc,
)
async def interview_agent(ctx: agents.JobContext) -> None:
    """
    LiveKit room entrypoint for a candidate interview session.
    Configures the agent with STT/TTS settings and starts the event loop.

    Args:
        ctx: The LiveKit job context.

    Raises:
        RuntimeError: If critical metadata is missing from the job.
    """

    metadata = json.loads(ctx.job.metadata or "{}")
    candidate_assessment_id = metadata.get("candidate_assessment_id")
    session_mode = str(metadata.get("session_mode") or "interview").lower()
    interview_session_id = str(metadata.get("interview_session_id") or "")
    connection_id = str(metadata.get("connection_id") or "")

    if not candidate_assessment_id:
        raise RuntimeError("Missing candidate_assessment_id in LiveKit job metadata")
    if session_mode not in {"interview", "demo"}:
        raise RuntimeError(f"Unsupported LiveKit session mode: {session_mode}")
    if session_mode == "interview" and (not interview_session_id or not connection_id):
        raise RuntimeError(
            "Missing interview session connection metadata in LiveKit job"
        )

    session: AgentSession = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(
            model=settings.DEEPGRAM_STT_MODEL or "nova-3",
            language="en",
            interim_results=True,
            punctuate=True,
            smart_format=True,
            no_delay=True,
            filler_words=True,
            vad_events=True,
            api_key=settings.DEEPGRAM_API_KEY or agents.NOT_GIVEN,
        ),
        tts=deepgram.TTS(
            model=settings.DEEPGRAM_TTS_MODEL or "aura-2-andromeda-en",
            api_key=settings.DEEPGRAM_API_KEY or None,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(
                version="v1",
                api_key=settings.LIVEKIT_API_KEY,
                api_secret=settings.LIVEKIT_API_SECRET,
            ),
            endpointing={
                "mode": "dynamic",
                "min_delay": settings.TURN_ENDPOINTING_MIN_DELAY_SECS,
                "max_delay": settings.TURN_ENDPOINTING_MAX_DELAY_SECS,
            },
            interruption={
                "enabled": True,
                "mode": "adaptive",
                "min_duration": 0.5,
                "min_words": 1,
                "false_interruption_timeout": (
                    settings.FALSE_INTERRUPTION_TIMEOUT_SECS
                ),
                "resume_false_interruption": True,
                "discard_audio_if_uninterruptible": True,
            },
            preemptive_generation={
                "enabled": settings.PREEMPTIVE_GENERATION_ENABLED,
                "preemptive_tts": False,
            },
        ),
        user_away_timeout=USER_AWAY_TIMEOUT_SECS,
    )

    agent: InterviewLiveKitAgent | DemoLiveKitAgent
    if session_mode == "demo":
        agent = DemoLiveKitAgent()
    else:
        agent = InterviewLiveKitAgent(
            candidate_assessment_id=str(candidate_assessment_id),
            interview_session_id=interview_session_id,
            connection_id=connection_id,
        )

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: agents.AgentStateChangedEvent) -> None:
        raw_agent_state = getattr(ev.new_state, "value", ev.new_state)
        if isinstance(agent, InterviewLiveKitAgent):
            agent.track_agent_state(str(raw_agent_state))

        logger.info(
            "LiveKit agent state changed",
            extra={
                "candidate_assessment_id": candidate_assessment_id,
                "session_mode": session_mode,
                "old_state": str(getattr(ev.old_state, "value", ev.old_state)),
                "new_state": str(getattr(ev.new_state, "value", ev.new_state)),
            },
        )

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: agents.UserStateChangedEvent) -> None:
        raw_state = getattr(ev.new_state, "value", ev.new_state)
        new_state = str(raw_state)

        if not isinstance(agent, InterviewLiveKitAgent):
            return

        agent.track_user_state(new_state)
        if new_state == "away":
            asyncio.create_task(agent.handle_user_away())

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: agents.UserInputTranscribedEvent) -> None:
        if isinstance(agent, InterviewLiveKitAgent):
            agent.track_user_transcription(
                ev.transcript or "",
                is_final=bool(ev.is_final),
            )
        logger.info(
            "LiveKit user input transcribed",
            extra={
                "candidate_assessment_id": candidate_assessment_id,
                "session_mode": session_mode,
                "is_final": ev.is_final,
                "text_length": len(ev.transcript or ""),
            },
        )

    @session.on("error")
    def _on_error(ev: Any) -> None:
        logger.error(
            "LiveKit AgentSession error",
            extra={
                "candidate_assessment_id": candidate_assessment_id,
                "session_mode": session_mode,
                "error": str(getattr(ev, "error", ev)),
            },
        )

    @session.on("close")
    def _on_close(ev: Any) -> None:
        reason = str(getattr(ev, "reason", ""))
        logger.info(
            "LiveKit AgentSession closed",
            extra={
                "candidate_assessment_id": candidate_assessment_id,
                "session_mode": session_mode,
                "reason": reason,
            },
        )
        if isinstance(agent, InterviewLiveKitAgent):
            asyncio.create_task(agent.handle_session_close(reason))

    rtc_config = (
        rtc.RtcConfiguration(
            ice_transport_type=rtc.IceTransportType.TRANSPORT_RELAY,
        )
        if settings.LIVEKIT_FORCE_RELAY
        else None
    )
    logger.info(
        "Connecting LiveKit job",
        extra={
            "candidate_assessment_id": candidate_assessment_id,
            "session_mode": session_mode,
            "transport_mode": (
                "relay" if settings.LIVEKIT_FORCE_RELAY else "automatic"
            ),
        },
    )
    await ctx.connect(
        auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY,
        rtc_config=rtc_config,
    )
    await ctx.wait_for_participant()

    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    agents.cli.run_app(server)
