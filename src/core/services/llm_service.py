"""
LLM service using Groq's chat completion API (non-streaming for this pass).

Maintains a per-session conversation history and returns the full
assistant reply as a string.
"""

from __future__ import annotations

import logging
from typing import Any

from groq import AsyncGroq

from src.config.settings import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are iBot, an intelligent AI interview assistant. "
    "Your role is to conduct a professional job interview by asking clear, relevant questions "
    "one at a time, evaluating candidate responses, and keeping a conversational yet professional tone. "
    "Keep your responses concise (2-4 sentences maximum) so they are natural in a voice conversation."
)


class LLMService:
    """
    Stateful LLM session that preserves conversation history for a single interview.

    Usage:
        llm = LLMService(candidate_id="...", assessment_id="...")
        reply = await llm.get_reply("Tell me about yourself.")
    """

    def __init__(self, candidate_id: str, assessment_id: str) -> None:
        self._candidate_id = candidate_id
        self._assessment_id = assessment_id
        self._client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self._history: list[dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
        ]

    async def get_reply(self, user_text: str) -> str:
        """
        Append user_text to conversation history, call Groq, and return the reply.

        Args:
            user_text: The candidate's transcribed or typed message.

        Returns:
            The assistant's reply text.
        """
        self._history.append({"role": "user", "content": user_text})

        try:
            completion = await self._client.chat.completions.create(
                model=settings.GROQ_MODEL,
                messages=self._history,  # type: ignore[arg-type]
                max_tokens=settings.GROQ_MAX_TOKENS,
                temperature=settings.GROQ_TEMPERATURE,
            )
            reply: str = completion.choices[0].message.content or ""
            self._history.append({"role": "assistant", "content": reply})
            logger.debug(
                "Groq reply for candidate=%s: %s",
                self._candidate_id,
                reply[:80],
            )
            return reply
        except Exception as exc:
            logger.exception("Groq LLM error for candidate=%s", self._candidate_id)
            raise RuntimeError(f"LLM call failed: {exc}") from exc

    def clear_history(self) -> None:
        """Reset conversation to just the system prompt."""
        self._history = [{"role": "system", "content": _SYSTEM_PROMPT}]
