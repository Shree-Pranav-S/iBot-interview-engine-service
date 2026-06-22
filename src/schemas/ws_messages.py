"""
WebSocket message contract for interview sessions.

Client → Server (JSON text frames):
    - session_start  : begin a new interview session
    - session_resume : explicitly request session reconnection
    - stop           : candidate ends the session
    - ping           : keep-alive

Client → Server (binary frames):
    - raw 16 kHz mono linear16 PCM audio chunk

Server → Client (JSON text frames):
    - connection_ack      : handshake confirmed, session id echoed
    - session_resumed     : reconnection confirmed with restored state
    - partial_transcript  : in-progress STT word hypothesis
    - final_transcript    : committed STT utterance
    - assistant_text      : bot reply text (for TTS + subtitles)
    - tts_audio           : signals that binary audio frame follows
    - section_start       : new section beginning
    - section_transition  : moving between sections
    - time_warning        : time reminder for section or overall
    - bot_speaking        : bot is delivering TTS audio
    - bot_done_speaking   : bot finished, mic is active
    - think_timer_start   : candidate think timer activated
    - interview_complete  : interview finished
    - session_terminated  : irrelevant strike out
    - session_deactivated : grace period expired
    - error               : error notification
    - pong                : keep-alive response
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ── Inbound (client → server) ─────────────────────────────────────────────────


class ClientMessageType(StrEnum):
    SESSION_START = "session_start"
    SESSION_RESUME = "session_resume"
    TEXT_MESSAGE = "text_message"
    STOP = "stop"
    PING = "ping"


class ClientMessage(BaseModel):
    """Generic envelope for text frames sent by the client."""

    type: ClientMessageType
    payload: dict[str, Any] = Field(default_factory=dict)


# ── Outbound (server → client) ────────────────────────────────────────────────


class ServerMessageType(StrEnum):
    CONNECTION_ACK = "connection_ack"
    SESSION_RESUMED = "session_resumed"
    PARTIAL_TRANSCRIPT = "partial_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
    ASSISTANT_TEXT = "assistant_text"
    TTS_AUDIO = "tts_audio"  # signals that binary frames follow
    ERROR = "error"
    PONG = "pong"

    # ── Interview-specific events ─────────────────────────────────────────
    SECTION_START = "section_start"
    SECTION_TRANSITION = "section_transition"
    TIME_WARNING = "time_warning"
    BOT_SPEAKING = "bot_speaking"
    BOT_DONE_SPEAKING = "bot_done_speaking"
    THINK_TIMER_START = "think_timer_start"
    INTERVIEW_COMPLETE = "interview_complete"
    SESSION_TERMINATED = "session_terminated"
    SESSION_DEACTIVATED = "session_deactivated"


class ServerMessage(BaseModel):
    """Generic envelope for text frames sent by the server."""

    type: ServerMessageType
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json()
