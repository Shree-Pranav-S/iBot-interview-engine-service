"""LiveKit voice agent worker for AI interviews.

Deepgram Nova-3 provides transcript quality, while LiveKit TurnDetector owns
end-of-turn detection before LangGraph receives a candidate response.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterable
from typing import Any

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
from livekit.plugins import deepgram, silero

from src.config.settings import settings
from src.core.services.livekit_graph_bridge import LiveKitInterviewBridge
from src.utils.livekit import chunk_for_tts

logger = logging.getLogger("interview-livekit-agent")


def prewarm(proc: agents.JobProcess) -> None:
    """Load expensive models once per worker process."""
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(
    ws_url=settings.LIVEKIT_URL or None,
    api_key=settings.LIVEKIT_API_KEY or None,
    api_secret=settings.LIVEKIT_API_SECRET or None,
    setup_fnc=prewarm,
)


class LangGraphPipelineLLM(llm.LLM):  # type: ignore[misc]
    """Placeholder LLM required for the AgentSession voice pipeline.

    The actual response generation happens in InterviewLiveKitAgent.llm_node.
    """

    @property
    def model(self) -> str:
        return "langgraph-interview"

    @property
    def provider(self) -> str:
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
        raise RuntimeError("LangGraphPipelineLLM is only used with a custom llm_node")


class InterviewLiveKitAgent(Agent):  # type: ignore[misc]
    """LiveKit voice agent that delegates interview reasoning to LangGraph."""

    def __init__(self, *, candidate_assessment_id: str) -> None:
        super().__init__(
            instructions=("placeholder"),
            llm=LangGraphPipelineLLM(),
        )

        self.bridge = LiveKitInterviewBridge(
            candidate_assessment_id=candidate_assessment_id
        )

        self._pending_user_text: str | None = None
        self._pending_duration_ms: int | None = None
        self._user_turn_started_at: float | None = None
        self._last_user_turn_duration_ms: int | None = None
        self._interview_closed = False
        self._turn_lock = asyncio.Lock()

    def _mark_closed_from_state(self) -> bool:
        state = self.bridge.state or {}
        self._interview_closed = bool(
            state.get("closing_done")
            or state.get("should_close")
            or state.get("bot_reply_type") == "closing"
        )
        return self._interview_closed

    def _shutdown_after_closing(self) -> None:
        try:
            self.session.shutdown(drain=True)
        except Exception:
            logger.exception(
                "Failed to shutdown LiveKit session after closing",
                extra={"candidate_assessment_id": self.bridge.candidate_assessment_id},
            )

    async def on_enter(self) -> None:
        """Start or resume the interview and speak the opening graph turn."""

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

        handle = self.session.say(
            opening_text,
            allow_interruptions=not self._interview_closed,
            add_to_chat_ctx=True,
        )
        await handle.wait_for_playout()
        if self._interview_closed:
            self._shutdown_after_closing()

    def track_user_state(self, new_state: str) -> None:
        """Record user speech duration for optional graph/timing metadata."""

        if new_state == "speaking" and self._user_turn_started_at is None:
            self._user_turn_started_at = time.monotonic()
            return

        if new_state == "speaking" or self._user_turn_started_at is None:
            return

        self._last_user_turn_duration_ms = int(
            (time.monotonic() - self._user_turn_started_at) * 1000
        )
        self._user_turn_started_at = None

    async def on_user_turn_completed(
        self,
        turn_ctx: ChatContext,
        new_message: ChatMessage,
    ) -> None:
        """Capture only LiveKit-confirmed user turns for graph submission."""

        if self._interview_closed:
            raise StopResponse()

        candidate_text = " ".join((new_message.text_content or "").split())

        if not candidate_text:
            raise StopResponse()

        if self._user_turn_started_at is not None:
            self.track_user_state("listening")

        self._pending_duration_ms = self._last_user_turn_duration_ms
        self._last_user_turn_duration_ms = None
        self._pending_user_text = candidate_text

    async def handle_user_away(self) -> None:
        """Resume the graph with its existing silence sentinel."""

        if self._interview_closed:
            return

        async with self._turn_lock:
            reply_text = await self.bridge.submit_silence()
            is_closing_reply = self._mark_closed_from_state()

        if not reply_text:
            return

        handle = self.session.say(
            reply_text,
            allow_interruptions=not is_closing_reply,
            add_to_chat_ctx=True,
        )
        await handle.wait_for_playout()
        if is_closing_reply:
            self._shutdown_after_closing()

    async def llm_node(
        self,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[str]:
        """Generate assistant text by resuming the existing LangGraph workflow."""

        candidate_text = self._pending_user_text
        duration_ms = self._pending_duration_ms

        self._pending_user_text = None
        self._pending_duration_ms = None

        if self._interview_closed:
            return

        if not candidate_text:
            return

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

        if not reply_text:
            return

        for chunk in chunk_for_tts(reply_text):
            yield chunk

        if is_closing_reply:
            self._shutdown_after_closing()


async def request_fnc(req: agents.JobRequest) -> None:
    """Accept explicitly dispatched interview-agent jobs."""

    await req.accept(
        name="AI Interview Bot",
        identity=f"interview-bot-{req.id}",
    )


@server.rtc_session(
    agent_name=settings.LIVEKIT_AGENT_NAME,
    on_request=request_fnc,
)
async def interview_agent(ctx: agents.JobContext) -> None:
    """LiveKit room entrypoint for a candidate interview session."""

    metadata = json.loads(ctx.job.metadata or "{}")
    candidate_assessment_id = metadata.get("candidate_assessment_id")

    if not candidate_assessment_id:
        raise RuntimeError("Missing candidate_assessment_id in LiveKit job metadata")

    session = AgentSession(
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
            api_key=settings.DEEPGRAM_API_KEY or agents.NOT_GIVEN,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            endpointing={
                "mode": "dynamic",
                "min_delay": 0.7,
                "max_delay": 5.0,
                "alpha": 0.85,
            },
            interruption={
                "enabled": True,
                "mode": "adaptive",
                "min_duration": 0.45,
                "min_words": 1,
                "false_interruption_timeout": 2.0,
                "resume_false_interruption": True,
                "discard_audio_if_uninterruptible": True,
            },
            preemptive_generation={"enabled": False},
        ),
        user_away_timeout=6.0,
    )

    agent = InterviewLiveKitAgent(candidate_assessment_id=str(candidate_assessment_id))

    @session.on("agent_state_changed")
    def _on_agent_state_changed(ev: agents.AgentStateChangedEvent) -> None:
        logger.info(
            "LiveKit agent state changed",
            extra={
                "candidate_assessment_id": candidate_assessment_id,
                "old_state": str(getattr(ev.old_state, "value", ev.old_state)),
                "new_state": str(getattr(ev.new_state, "value", ev.new_state)),
            },
        )

    @session.on("user_state_changed")
    def _on_user_state_changed(ev: agents.UserStateChangedEvent) -> None:
        raw_state = getattr(ev.new_state, "value", ev.new_state)
        new_state = str(raw_state)

        agent.track_user_state(new_state)

        if new_state == "away":
            asyncio.create_task(agent.handle_user_away())

    @session.on("user_input_transcribed")
    def _on_user_input_transcribed(ev: agents.UserInputTranscribedEvent) -> None:
        logger.info(
            "LiveKit user input transcribed",
            extra={
                "candidate_assessment_id": candidate_assessment_id,
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
                "error": str(getattr(ev, "error", ev)),
            },
        )

    @session.on("close")
    def _on_close(ev: Any) -> None:
        logger.info(
            "LiveKit AgentSession closed",
            extra={
                "candidate_assessment_id": candidate_assessment_id,
                "reason": str(getattr(ev, "reason", "")),
            },
        )

    logger.info(
        "Connecting LiveKit job with relay transport",
        extra={"candidate_assessment_id": candidate_assessment_id},
    )
    await ctx.connect(
        auto_subscribe=agents.AutoSubscribe.AUDIO_ONLY,
        rtc_config=rtc.RtcConfiguration(
            ice_transport_type=rtc.IceTransportType.Value("TRANSPORT_RELAY"),
        ),
        single_peer_connection=True,
    )

    await session.start(agent=agent, room=ctx.room)


if __name__ == "__main__":
    agents.cli.run_app(server)
