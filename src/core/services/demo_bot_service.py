"""
Demo bot service providing static responses.

Simulates the LLM conversation loop for the demo/practice room.
"""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)


class DemoBotService:
    """
    Demo bot session that provides static replies from a preset list
    to simulate an interview.
    """

    def __init__(self, candidate_id: str, assessment_id: str) -> None:
        self._candidate_id = candidate_id
        self._assessment_id = assessment_id
        self._responses = [
            "Great. Could you tell me a bit about your experience working with web frameworks?",
            "That makes sense. How do you usually handle state management in a complex single-page application?",
            "Interesting approach. What is your strategy for optimizing the performance of slow-loading pages?",
            "I see. Have you worked with real-time communications like WebSockets before? If so, what was the use case?",
            "Thank you for sharing that. How do you approach writing clean, maintainable unit and integration tests?",
            "Got it. Can you explain the difference between client-side rendering and server-side rendering?",
            "That is a very common scenario. How do you handle conflict or differing technical opinions within your engineering team?",
            "Perfect. Could you describe a challenging technical bug you solved recently and what your process was?",
            "Excellent. What security practices do you keep in mind when designing frontend applications and APIs?",
            "That is good to know. How do you stay updated with the latest trends and updates in the software development ecosystem?",
            "Understood. Can you describe your experience with CI/CD pipelines and modern containerization technologies?",
            "Makes sense. How do you manage database schema migrations in a high-traffic production environment?",
            "Very clear. What would you say is your favorite programming language, and what makes it special to you?",
            "That is a solid answer. How do you prioritize tasks when you have multiple competing deadlines?",
            "Thank you. Finally, what questions do you have for us regarding the role or the team?",
        ]
        self._used_responses: list[str] = []

    async def get_reply(self, user_text: str) -> str:
        """Return a random question from the bank, avoiding immediate duplication."""
        # Find unused questions
        available = [r for r in self._responses if r not in self._used_responses]
        if not available:
            # Reset history if all are used
            self._used_responses.clear()
            available = self._responses

        reply = random.choice(available)
        self._used_responses.append(reply)

        logger.debug(
            "DemoBot static reply for candidate=%s: %s",
            self._candidate_id,
            reply[:80],
        )
        return reply

    def clear_history(self) -> None:
        """Reset the used question history."""
        self._used_responses.clear()
