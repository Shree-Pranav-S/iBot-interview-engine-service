"""
generate_question_node — Generates the next interview question.

The most critical node for interview naturalness. The prompt is designed
to make the bot indistinguishable from a seasoned human interviewer:

1. Conversational personality — acknowledges the previous answer before
   asking the next question (bridging). Never sounds robotic.
2. Adaptive difficulty — escalates on consecutive strong answers,
   recalibrates down on weak ones.
3. Follow-up intelligence — probes deeper if the previous answer was
   strong, tries a simpler angle if it was weak.
4. Time awareness — quick wrap-up questions under 60s, no deep dives
   under 120s.
5. Anti-repetition — never revisits a concept already covered.
6. Section context — first question in a section gets a warm transition;
   subsequent questions flow from the conversation.
"""

from __future__ import annotations

import json
import logging

from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_fixed

from src.config.settings import settings
from src.control.agents.state import InterviewState, SectionState

logger = logging.getLogger(__name__)


# ── System prompt: shapes the LLM into a specific interviewer persona ─────────

_SYSTEM_PROMPT = """\
You are a senior technical interviewer with 10+ years of hiring experience at \
top-tier tech companies. You are conducting a live voice interview — your words \
will be spoken aloud via text-to-speech, so you must sound completely natural.

Personality traits:
- Warm but professional. You put candidates at ease.
- You ALWAYS briefly acknowledge the candidate's previous answer before \
asking the next question. This is non-negotiable — never jump straight to \
a new question without a transition.
- You vary your acknowledgments. Mix short reactions \
("That's a solid point.", "Interesting.", "Right, I see what you mean.", \
"Good example.") with slightly longer ones that reference what they said.
- You occasionally use conversational filler that real interviewers use: \
"So...", "Let me ask you about...", "Alright, shifting gears a bit...", \
"Building on what you just said..."
- You NEVER use bullet points, markdown, or numbered lists.
- You NEVER say "Great answer!" or give away your evaluation.
- Keep each question concise — 1-3 sentences maximum. This is spoken aloud.
- Ask ONE question at a time. Never ask compound questions.
- Do NOT start with "Great question" or "That's a great question" — \
  you are the one asking questions, not answering them.

Output format: Return ONLY valid JSON (no markdown, no code fences):
{{"question_text": "Your full spoken utterance including the bridge/acknowledgment AND the question", "concept_tag": "the_core_concept_being_tested", "difficulty": "easy|medium|hard"}}
"""


# ── User prompt template: full context injection ──────────────────────────────

_QUESTION_PROMPT = """\
INTERVIEW CONTEXT
=================
Role: {role_title}
Current section: {section_name} (Skill: {skill}, Priority: {priority}/10)
Time remaining in this section: {time_remaining} seconds

JD key skills: {jd_skills}
Candidate background: {resume_summary}

CONVERSATION HISTORY (last {history_count} exchanges)
======================
{conversation_history}

EVALUATION OF LAST ANSWER
=========================
Quality: {quality}
Score: {score}/10
Signals demonstrated: {signals_present}
Signals still missing: {signals_missing}
Was it substantial enough to evaluate: {is_substantial}
Feedback: {feedback}

INTERVIEW PROGRESS
==================
Questions asked in this section: {questions_in_section}
Concepts already covered (DO NOT repeat): {used_concepts}
Consecutive strong answers: {consecutive_strong}
Consecutive weak answers: {consecutive_weak}

DIFFICULTY LEVEL: {current_difficulty}
{difficulty_reason}

SPECIAL INSTRUCTIONS
====================
{special_instructions}

Generate your next spoken utterance — remember to include a brief, natural \
acknowledgment of their last answer before asking the question. Return valid JSON only."""


# ── Helper functions ──────────────────────────────────────────────────────────


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def _call_groq_question(messages: list[dict]) -> str:
    """Call Groq with retry-backoff for rate limiting."""
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    completion = await client.chat.completions.create(  # type: ignore[call-overload]
        model=settings.GROQ_MODEL,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=settings.GROQ_MAX_TOKENS,
        temperature=settings.GROQ_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content or ""


def _determine_difficulty(
    consecutive_strong: int,
    consecutive_weak: int,
    priority: int,
) -> tuple[str, str]:
    """
    Adaptive difficulty based on recent performance.

    Returns (difficulty_level, reason_for_prompt).
    """
    if consecutive_strong >= 2:
        if priority <= 4:
            return "medium", (
                f"Candidate showing strength but skill priority is low "
                f"({priority}/10) — stay at medium, don't over-invest."
            )
        return "hard", (
            f"Escalate difficulty: {consecutive_strong} consecutive strong "
            f"answers. Ask about edge cases, failure modes, or system "
            f"design trade-offs."
        )
    elif consecutive_weak >= 2:
        return "easy", (
            f"Recalibrate down: {consecutive_weak} consecutive weak answers. "
            f"Ask a simpler, more foundational question. Give them a chance "
            f"to demonstrate basic understanding."
        )
    else:
        return "medium", "Standard difficulty — mix of conceptual and applied."


def _build_conversation_history(state: InterviewState) -> tuple[str, int]:
    """
    Build a formatted string of the last few Q&A exchanges
    from the transcript_turns list for context.

    Returns (formatted_history, turn_count).
    """
    turns = state.get("transcript_turns", [])
    if not turns:
        return "No conversation yet — this is the first question in this section.", 0

    # Take the last 4 turns (typically 2 Q&A pairs)
    recent = turns[-4:]
    lines: list[str] = []
    for turn in recent:
        speaker = turn.get("speaker", "?")
        text = turn.get("text", "")
        turn_type = turn.get("type", "")

        if speaker == "bot":
            label = "INTERVIEWER"
            if turn_type == "opening":
                label = "INTERVIEWER (opening)"
        else:
            label = "CANDIDATE"

        # Truncate very long candidate answers for prompt efficiency
        display_text = text[:300] + "..." if len(text) > 300 else text
        lines.append(f"{label}: {display_text}")

    return "\n".join(lines), len([t for t in recent if t.get("speaker") == "bot"])


def _build_special_instructions(
    state: InterviewState,
    current_section: SectionState | None,
    time_remaining: int,
    is_first_question_in_section: bool,
    last_eval: dict,
) -> str:
    """
    Build context-specific instructions that guide the LLM's behavior
    based on the current interview situation.
    """
    instructions: list[str] = []

    # First question in a section
    if is_first_question_in_section:
        section_name = current_section["name"] if current_section else "this section"
        instructions.append(
            f"This is the FIRST question in the '{section_name}' section. "
            f"Start with a natural transition like 'Alright, let's talk about...' "
            f"or 'So, moving on to {section_name}...'. Ask an opening question "
            f"that's approachable — don't start with the hardest thing."
        )

    # Time pressure
    if time_remaining < 60:
        instructions.append(
            "TIME IS ALMOST UP for this section (< 60 seconds). Ask a quick, "
            "focused wrap-up question. No deep dives. Something like "
            "'One last quick one on this topic...' or "
            "'Before we move on, can you briefly tell me...'"
        )
    elif time_remaining < 120:
        instructions.append(
            "Time is getting short (< 2 minutes). Keep the question focused. "
            "Don't open up a big new topic area."
        )

    # Follow-up intelligence based on evaluation
    quality = last_eval.get("quality", "")
    if quality == "strong":
        signals_missing = last_eval.get("signals_missing", [])
        if signals_missing:
            instructions.append(
                f"The candidate gave a strong answer. Probe deeper on these "
                f"missing signals: {', '.join(signals_missing)}. Ask a "
                f"follow-up that builds on what they said."
            )
        else:
            instructions.append(
                "The candidate gave a strong, comprehensive answer. "
                "Move to a new concept area — don't keep probing the same topic."
            )
    elif quality == "weak":
        instructions.append(
            "The candidate's last answer was weak. Try approaching the same "
            "skill area from a different, simpler angle. Maybe ask for a "
            "concrete example from their experience, or rephrase as a "
            "'have you ever encountered...' scenario."
        )
    elif quality == "non_answer":
        instructions.append(
            "The candidate couldn't answer the last question. Move to a "
            "completely different concept within the same skill. Don't dwell "
            "on what they couldn't answer."
        )
    elif quality == "adequate":
        is_substantial = last_eval.get("is_substantial", True)
        if not is_substantial:
            instructions.append(
                "The candidate's answer was too brief to properly evaluate. "
                "Ask a probing follow-up like 'Could you walk me through a "
                "specific example?' or 'Can you elaborate on that a bit more?'"
            )

    if not instructions:
        instructions.append(
            "Continue the conversation naturally. Ask about a new concept "
            "within the current skill area."
        )

    return "\n".join(f"- {i}" for i in instructions)


def _parse_question(raw: str) -> dict:
    """Parse the LLM question generation response, handling edge cases."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse question JSON. Raw output: {raw}")
        raise e

    if "question_text" not in result or not result["question_text"]:
        result["question_text"] = (
            "That's a good point. Could you walk me through a specific "
            "example from your experience?"
        )

    # Ensure difficulty is valid
    if result.get("difficulty") not in ("easy", "medium", "hard"):
        result["difficulty"] = "medium"

    return result


# ── Main node ─────────────────────────────────────────────────────────────────


async def generate_question_node(state: InterviewState) -> dict:
    """
    Generate the next interview question with full context awareness.

    This is the most important node for interview naturalness.
    The LLM receives rich context including conversation history,
    evaluation signals, and special situational instructions.
    """
    sections = state.get("sections", [])
    section_idx = state.get("current_section_index", 0)
    used_concepts = state.get("used_concepts", [])
    consecutive_strong = state.get("consecutive_strong", 0)
    consecutive_weak = state.get("consecutive_weak", 0)
    turn_number = state.get("turn_number", 0)
    plan = state.get("interview_plan", {})
    jd = state.get("jd_analysis", {})
    resume = state.get("resume_context", {})

    current_section = sections[section_idx] if section_idx < len(sections) else None
    skill = current_section["skill"] if current_section else "general"
    priority = current_section["priority_score"] if current_section else 5
    section_name = current_section["name"] if current_section else "General"
    time_remaining = state.get("current_section_time_remaining_secs", 300)
    questions_in_section = (
        current_section.get("questions_asked", 0) if current_section else 0
    )
    is_first_question = questions_in_section == 0

    # Evaluation context from the last answer
    last_eval = state.get("last_evaluation") or {}

    # Build conversation history
    conversation_history, _ = _build_conversation_history(state)

    # Adaptive difficulty
    difficulty, difficulty_reason = _determine_difficulty(
        consecutive_strong, consecutive_weak, priority
    )

    # Special situational instructions
    special_instructions = _build_special_instructions(
        state, current_section, time_remaining, is_first_question, last_eval
    )

    # Build the full user prompt
    prompt = _QUESTION_PROMPT.format(
        role_title=plan.get("role_title", "Software Engineer"),
        section_name=section_name,
        skill=skill,
        priority=priority,
        time_remaining=time_remaining,
        jd_skills=", ".join(jd.get("key_skills", ["general"])),
        resume_summary=resume.get("summary", "Not available"),
        history_count=min(3, turn_number),
        conversation_history=conversation_history,
        quality=last_eval.get("quality", "N/A (first question)"),
        score=last_eval.get("score", "N/A"),
        signals_present=(
            ", ".join(last_eval.get("signals_present", [])) or "none identified"
        ),
        signals_missing=(", ".join(last_eval.get("signals_missing", [])) or "none"),
        is_substantial=last_eval.get("is_substantial", "N/A"),
        feedback=last_eval.get("one_line_feedback", "N/A (first question)"),
        questions_in_section=questions_in_section,
        used_concepts=(", ".join(used_concepts) if used_concepts else "none yet"),
        consecutive_strong=consecutive_strong,
        consecutive_weak=consecutive_weak,
        current_difficulty=difficulty,
        difficulty_reason=difficulty_reason,
        special_instructions=special_instructions,
    )

    raw_response = await _call_groq_question(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
    )
    question_data = _parse_question(raw_response)

    question_text = question_data["question_text"]
    concept_tag = question_data.get("concept_tag", "")

    logger.info(
        "Question generated: turn=%d difficulty=%s concept=%s",
        turn_number,
        question_data.get("difficulty", "medium"),
        concept_tag,
    )

    # Update section questions count
    if current_section:
        updated_sections = list(sections)
        covered = list(current_section.get("concepts_covered", []))
        if concept_tag and concept_tag not in covered:
            covered.append(concept_tag)

        updated_section: SectionState = {
            "name": current_section.get("name", ""),
            "skill": current_section.get("skill", ""),
            "priority_score": current_section.get("priority_score", 5),
            "time_budget_secs": current_section.get("time_budget_secs", 300),
            "time_elapsed_secs": current_section.get("time_elapsed_secs", 0),
            "questions_asked": current_section.get("questions_asked", 0) + 1,
            "concepts_covered": covered,
            "is_complete": current_section.get("is_complete", False),
        }
        updated_sections[section_idx] = updated_section
    else:
        updated_sections = sections

    # Record bot question turn
    turn_record = {
        "turn_number": turn_number,
        "speaker": "bot",
        "text": question_text,
        "section": section_name,
        "type": "question",
        "concept_tag": concept_tag,
        "difficulty": question_data.get("difficulty", "medium"),
    }

    return {
        "current_question_text": question_text,
        "current_question_difficulty": question_data.get("difficulty", "medium"),
        "bot_reply_text": question_text,
        "bot_reply_type": "question",
        "sections": updated_sections,
        "transcript_turns": state.get("transcript_turns", []) + [turn_record],
        # Reset silence attempt counter on new question
        "silence_attempt": 0,
    }
