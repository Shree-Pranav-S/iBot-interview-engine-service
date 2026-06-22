"""STT (Speech-to-Text) service using Deepgram Nova-3 streaming."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from src.config.settings import settings

logger = logging.getLogger(__name__)


class TranscriptEvent:
    """Carries a single transcript result from Deepgram."""

    def __init__(
        self,
        text: str,
        is_final: bool,
        event_type: str = "transcript",
        confidence: float = 1.0,
    ) -> None:
        self.text = text
        self.is_final = is_final
        self.event_type = event_type
        self.confidence = confidence


class STTService:
    def __init__(
        self,
        encoding: str | None = None,
        sample_rate: str | int | None = None,
    ) -> None:
        self._client = AsyncDeepgramClient(api_key=settings.DEEPGRAM_API_KEY)
        self._connection = None
        self._connection_cm = None
        self._listen_task: asyncio.Task | None = None
        self._queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        self._encoding = encoding
        self._sample_rate = sample_rate
        self._closed = False

    async def __aenter__(self) -> STTService:
        await self._connect()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def _connect(self) -> None:
        stt_model = (settings.DEEPGRAM_STT_MODEL or "nova-3").strip()
        options: dict[str, str | int | bool] = {
            "model": stt_model,
            "interim_results": True,
            "vad_events": True,
            "punctuate": True,
            "smart_format": True,
            "language": "en",
            "endpointing": 500,
            "utterance_end_ms": 3000,
        }
        if self._encoding:
            options["encoding"] = self._encoding
        if self._sample_rate:
            options["sample_rate"] = self._sample_rate
        self._connection_cm = self._client.listen.v1.connect(**options)
        self._connection = await self._connection_cm.__aenter__()

        def on_message(*args) -> None:
            try:
                message = args[1] if len(args) > 1 else args[0]
                message_type = self._get_field(message, "type", "")

                if message_type in {"Error", "ConfigureFailure"}:
                    code = self._get_field(message, "code", "")
                    description = self._get_field(message, "description", "")
                    logger.error(
                        "Deepgram Nova %s: %s %s",
                        message_type,
                        code,
                        description,
                    )
                    self._closed = True
                    self._queue.put_nowait(None)
                    return

                if message_type == "SpeechStarted":
                    self._queue.put_nowait(
                        TranscriptEvent(
                            text="",
                            is_final=False,
                            event_type="speech_started",
                        )
                    )
                    return

                if message_type == "UtteranceEnd":
                    self._queue.put_nowait(
                        TranscriptEvent(
                            text="",
                            is_final=True,
                            event_type="utterance_end",
                        )
                    )
                    return

                if message_type != "Results":
                    return

                transcript, confidence = self._extract_results(message)
                is_final = bool(self._get_field(message, "is_final", False))
                speech_final = bool(self._get_field(message, "speech_final", False))

                if not transcript.strip():
                    if speech_final:
                        self._queue.put_nowait(
                            TranscriptEvent(
                                text="",
                                is_final=True,
                                event_type="utterance_end",
                                confidence=confidence,
                            )
                        )
                    return

                if speech_final:
                    event_type = "speech_final"
                elif is_final:
                    event_type = "segment_final"
                else:
                    event_type = "transcript"

                logger.debug(
                    "Deepgram Nova STT: event=%s final=%s speech_final=%s text=%r",
                    event_type,
                    is_final,
                    speech_final,
                    transcript,
                )
                self._queue.put_nowait(
                    TranscriptEvent(
                        text=transcript,
                        is_final=is_final or speech_final,
                        event_type=event_type,
                        confidence=confidence,
                    )
                )
            except Exception:
                logger.exception("Error handling Deepgram Nova message")

        async def on_error(error) -> None:
            logger.error("Deepgram connection error: %s", error)

        self._connection.on(EventType.MESSAGE, on_message)
        self._connection.on(EventType.ERROR, on_error)

        self._listen_task = asyncio.create_task(self._connection.start_listening())

        logger.info(
            "Deepgram Nova STT connected (model=%s encoding=%s sample_rate=%s)",
            stt_model,
            self._encoding or "auto",
            self._sample_rate or "auto",
        )

    @staticmethod
    def _get_field(message: Any, field: str, default: Any = None) -> Any:
        if isinstance(message, dict):
            return message.get(field, default)
        return getattr(message, field, default)

    @classmethod
    def _extract_results(cls, message: Any) -> tuple[str, float]:
        channel = cls._get_field(message, "channel", None)
        alternatives = (
            cls._get_field(channel, "alternatives", []) if channel is not None else []
        )
        if not alternatives:
            return "", 1.0

        alternative = alternatives[0]
        transcript = cls._get_field(alternative, "transcript", "") or ""
        confidence = cls._get_field(alternative, "confidence", 1.0)
        if confidence is None:
            confidence = 1.0
        return transcript, float(confidence)

    async def send_audio(self, data: bytes) -> bool:
        if self._connection is None or self._closed:
            return False

        try:
            await self._connection.send_media(data)
            return True
        except ConnectionClosedOK:
            logger.debug("Deepgram STT connection already closed while sending audio")
        except ConnectionClosed:
            logger.info("Deepgram STT connection closed while sending audio")
        except Exception:
            logger.exception("Error sending audio to Deepgram STT")
        self._closed = True
        return False

    async def transcript_events(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event

    async def close(self) -> None:
        if self._connection is not None:
            try:
                await self._connection.send_close_stream()
            except ConnectionClosedOK:
                logger.debug("Deepgram STT connection already closed")
            except ConnectionClosed:
                logger.info("Deepgram STT connection closed before close stream")
            except Exception:
                logger.exception("Error closing Deepgram STT connection")
            finally:
                self._closed = True

        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass

        if self._connection_cm is not None:
            try:
                await self._connection_cm.__aexit__(None, None, None)
            except Exception:
                logger.exception("Error exiting Deepgram connection context")
            finally:
                self._connection_cm = None
                self._connection = None

        await self._queue.put(None)
