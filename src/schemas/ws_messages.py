"""
WebSocket message contract for interview sessions.

Client → Server (JSON text frames):
    - session_start  : begin a new interview session
    - text_message   : typed text input from the candidate
    - stop           : candidate ends the session

Client → Server (binary frames):
    - raw PCM/opus audio chunk (prefixed with "audio:" is NOT used here;
      binary frames are treated as raw audio automatically)

Server → Client (JSON text frames):
    - connection_ack      : handshake confirmed, session id echoed
    - partial_transcript  : in-progress STT word hypothesis
    - final_transcript    : committed STT utterance
    - assistant_text      : LLM reply (streamed or complete)
    - tts_audio           : binary audio chunk (sent as binary frame)
    - error               : error notification
    - ping / pong         : keep-alive
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ── Inbound (client → server) ─────────────────────────────────────────────────


class ClientMessageType(StrEnum):
    SESSION_START = "session_start"
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
    PARTIAL_TRANSCRIPT = "partial_transcript"
    FINAL_TRANSCRIPT = "final_transcript"
    ASSISTANT_TEXT = "assistant_text"
    TTS_AUDIO = "tts_audio"  # signals that binary frames follow
    ERROR = "error"
    PONG = "pong"


class ServerMessage(BaseModel):
    """Generic envelope for text frames sent by the server."""

    type: ServerMessageType
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> str:
        return self.model_dump_json()
