"""
classify_response_node — Central router hub of the interview graph.

LLM-based classifier that categorises the candidate's transcript
into one of: SILENCE, CLARIFICATION, IRRELEVANT, or ANSWER.
The conditional edge from this node dispatches to the appropriate handler.
"""

from __future__ import annotations

import logging
from datetime import UTC

from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_fixed

from src.config.settings import settings
from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)

# ── LLM classification prompts ───────────────────────────────────────────────

_CLASSIFY_SYSTEM_PROMPT = """\
You are an expert technical interviewer assistant. Your task is to classify a candidate's response into one of the following categories:

1. "clarification": The candidate is asking to repeat, rephrase, explain, or clarify the question, or expressing confusion about what is being asked.
   Examples:
   - "Could you please repeat the question?"
   - "What do you mean by that?"
   - "I'm not sure I understand the question."

2. "irrelevant": The candidate's response is completely off-topic, completely unrelated to the question or the skill being assessed, or is talking about something else entirely.
   Examples:
   - "Oh, look at the weather today."
   - "Can we talk about something else?"
   - "I like pizza."

3. "answer": The candidate is attempting to answer the question (whether the answer is correct, partial, short, or weak).
   Examples:
   - "Yes, a database index is used to speed up queries."
   - "I don't know the exact answer, but I think it relates to caching."
   - "Python is interpreted."

Given the current skill, the question asked, and the candidate's transcript, classify the response.
Return ONLY one of the following lowercase strings: "clarification", "irrelevant", or "answer". Do not include any other text, punctuation, or formatting.
"""

_CLASSIFY_USER_TEMPLATE = """\
Skill: {skill}
Question asked: {question}
Candidate transcript: "{transcript}"

Classification:"""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def _call_groq_classify(messages: list[dict]) -> str:
    """Call Groq with retry-backoff for rate limiting."""
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    completion = await client.chat.completions.create(
        model=settings.GROQ_CLASSIFY_MODEL,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=settings.GROQ_CLASSIFY_MAX_TOKENS,
        temperature=settings.GROQ_CLASSIFY_TEMPERATURE,
    )
    return completion.choices[0].message.content or ""


async def classify_response_node(state: InterviewState) -> dict:
    """
    Classify the candidate's last transcript using LLM and set the classification field.

    Returns the classification label which the graph's conditional edge
    uses to route to the appropriate handler node.
    """
    from datetime import datetime

    transcript = state.get("last_transcript", "").strip()
    question = state.get("current_question_text", "")
    sections = state.get("sections", [])
    section_idx = state.get("current_section_index", 0)
    current_skill = (
        sections[section_idx]["skill"] if section_idx < len(sections) else "general"
    )

    # ── Update time tracking dynamically ─────────────────────────────────────
    now = datetime.now(UTC)
    started_at_str = state.get("timer_started_at")
    total_pause = state.get("total_pause_secs", 0)

    if started_at_str:
        try:
            started_at = datetime.fromisoformat(started_at_str)
            total_elapsed = int((now - started_at).total_seconds()) - total_pause
            total_elapsed = max(0, total_elapsed)
        except Exception:
            logger.exception("Failed to parse timer_started_at")
            total_elapsed = state.get("total_elapsed_secs", 0)
    else:
        total_elapsed = state.get("total_elapsed_secs", 0)

    completed_elapsed = 0
    for idx, sec in enumerate(sections):
        if idx < section_idx:
            completed_elapsed += sec.get("time_elapsed_secs", 0)

    current_sec = sections[section_idx] if section_idx < len(sections) else None
    if current_sec:
        budget = current_sec.get("time_budget_secs", 300)
        current_elapsed = total_elapsed - completed_elapsed
        remaining = budget - current_elapsed
    else:
        remaining = 0

    time_updates = {
        "total_elapsed_secs": total_elapsed,
        "current_section_time_remaining_secs": remaining,
    }

    # 0. Deterministic TIME_UP check
    if transcript == "__TIME_UP__":
        logger.info("Classification: TIME_UP (deterministic)")
        return {
            "last_response_classification": "time_up",
            **time_updates,
        }

    # 1. Deterministic SILENCE check (avoid calling LLM for empty/silence responses)
    if not transcript or transcript == "__SILENCE__":
        classification = "silence"
        logger.info("Classification: SILENCE (deterministic)")
        return {
            "last_response_classification": classification,
            **time_updates,
        }

    # 2. LLM-based classification for clarification, irrelevant, answer
    prompt = _CLASSIFY_USER_TEMPLATE.format(
        skill=current_skill,
        question=question,
        transcript=transcript,
    )

    try:
        raw_response = await _call_groq_classify(
            [
                {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        cleaned = raw_response.strip().strip("'\"`.,").lower()

        # Validate category
        if cleaned in ("clarification", "irrelevant", "answer"):
            classification = cleaned
        else:
            # Check if keyword is inside response
            if "clarification" in cleaned:
                classification = "clarification"
            elif "irrelevant" in cleaned:
                classification = "irrelevant"
            elif "answer" in cleaned:
                classification = "answer"
            else:
                logger.warning(
                    "Unexpected classification response: '%s', defaulting to 'answer'",
                    raw_response,
                )
                classification = "answer"
    except Exception:
        logger.exception("Classification LLM call failed, defaulting to 'answer'")
        classification = "answer"

    logger.info("Classification: %s", classification.upper())
    return {
        "last_response_classification": classification,
        **time_updates,
    }


def route_response(state: InterviewState) -> str:
    """
    Conditional edge function: routes from classify_response to the
    appropriate handler node based on classification.
    """
    classification = state.get("last_response_classification", "answer")
    logger.debug("Routing based on classification: %s", classification)
    return classification
