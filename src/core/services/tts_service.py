"""
TTS (Text-to-Speech) service using Deepgram's REST synthesis endpoint.

Returns raw MP3 audio bytes for a given text string.
Called once per complete LLM response to produce the voice reply.
"""

from __future__ import annotations

import logging

import httpx

from src.config.settings import settings

logger = logging.getLogger(__name__)

_DEEPGRAM_TTS_URL = "https://api.deepgram.com/v1/speak"


async def synthesize_speech(text: str) -> bytes:
    """
    Call the Deepgram TTS REST endpoint and return MP3 audio bytes.

    Args:
        text: The assistant response text to convert to speech.

    Returns:
        Raw MP3 bytes ready to be sent as a binary WebSocket frame.

    Raises:
        RuntimeError: If the Deepgram API returns an error.
    """
    if not settings.DEEPGRAM_API_KEY:
        raise RuntimeError("DEEPGRAM_API_KEY is not configured.")

    headers = {
        "Authorization": f"Token {settings.DEEPGRAM_API_KEY}",
        "Content-Type": "application/json",
    }
    params = {"model": settings.DEEPGRAM_TTS_MODEL}
    body = {"text": text}

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            _DEEPGRAM_TTS_URL,
            headers=headers,
            params=params,
            json=body,
        )

    if response.status_code != 200:
        logger.error(
            "Deepgram TTS error: status=%d body=%s",
            response.status_code,
            response.text[:200],
        )
        raise RuntimeError(f"Deepgram TTS failed with status {response.status_code}")

    logger.debug(
        "Deepgram TTS synthesized %d bytes for %d chars",
        len(response.content),
        len(text),
    )
    return response.content
