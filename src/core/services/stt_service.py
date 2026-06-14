"""
STT (Speech-to-Text) service using Deepgram streaming.

Streams raw audio bytes from the client into Deepgram's live
transcription API and yields transcript events back to the caller.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from deepgram import (
    DeepgramClient,
    DeepgramClientOptions,
    LiveOptions,
    LiveTranscriptionEvents,
)

from src.config.settings import settings

logger = logging.getLogger(__name__)


class TranscriptEvent:
    """Carries a single transcript result from Deepgram."""

    def __init__(self, text: str, is_final: bool) -> None:
        self.text = text
        self.is_final = is_final


class STTService:
    """
    Wraps the Deepgram async live-transcription connection for a single session.

    Usage:
        async with STTService() as stt:
            stt.send_audio(chunk)
            async for event in stt.transcript_events():
                ...
    """

    def __init__(self) -> None:
        config = DeepgramClientOptions(options={"keepalive": True})
        self._client = DeepgramClient(settings.DEEPGRAM_API_KEY, config=config)
        self._connection = None
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()

    async def __aenter__(self) -> STTService:
        await self._connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _connect(self) -> None:
        options = LiveOptions(
            model=settings.DEEPGRAM_STT_MODEL,
            language="en-US",
            smart_format=True,
            interim_results=True,
            endpointing=500,
        )

        self._connection = self._client.listen.asyncwebsocket.v("1")

        async def on_transcript(self_ref: object, result: object, **_: object) -> None:  # type: ignore[misc]
            try:
                alternatives = result.channel.alternatives  # type: ignore[attr-defined]
                if not alternatives:
                    return
                transcript = alternatives[0].transcript
                if not transcript:
                    return
                is_final: bool = result.is_final  # type: ignore[attr-defined]
                await self._queue.put(
                    TranscriptEvent(text=transcript, is_final=is_final)
                )
            except Exception:
                logger.exception("Error handling Deepgram transcript callback")

        self._connection.on(LiveTranscriptionEvents.Transcript, on_transcript)  # type: ignore[arg-type]
        started = await self._connection.start(options)
        if not started:
            raise RuntimeError("Failed to start Deepgram live connection")
        logger.info("Deepgram STT connection established")

    async def send_audio(self, data: bytes) -> None:
        """Push a raw audio chunk into the Deepgram stream."""
        if self._connection is not None:
            await self._connection.send(data)

    async def transcript_events(self) -> AsyncIterator[TranscriptEvent]:
        """Yield transcript events until the stream is closed (None sentinel)."""
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event

    async def close(self) -> None:
        """Finish sending and close the Deepgram connection."""
        if self._connection is not None:
            try:
                await self._connection.finish()
            except Exception:
                logger.exception("Error closing Deepgram STT connection")
            finally:
                self._connection = None
        # Signal iterator to stop
        await self._queue.put(None)
