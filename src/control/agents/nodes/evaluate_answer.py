"""
evaluate_answer_node — Live evaluation of candidate answers.

Makes an LLM call to evaluate the candidate's response against the
current question context, skill, and priority. Returns structured
JSON with quality, score, and signal analysis.
"""

from __future__ import annotations

import json
import logging

from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_fixed

from src.config.settings import settings
from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)

_EVALUATION_PROMPT = """\
You are an expert technical interview evaluator. Evaluate the candidate's answer.

Context:
- Skill being assessed: {skill}
- Priority: {priority}/10
- Question asked: {question_text}
{difficulty_context}
- Candidate answer: {transcript}
- Previously covered concepts: {used_concepts}

Evaluate this answer. Return ONLY valid JSON (no markdown, no code fences):
{{
  "quality": "strong|adequate|weak|non_answer",
  "score": <0-10>,
  "signals_present": ["signal1", "signal2"],
  "signals_missing": ["signal1"],
  "is_substantial": true|false,
  "key_concept_demonstrated": "concept or empty string",
  "one_line_feedback": "Brief internal note about the answer quality"
}}

Scoring guide:
- 0: No relevant answer / silence / refusal

{scoring_instructions}

is_substantial should be false if the answer is too brief (< 2 sentences) or
vague to properly evaluate the skill.
"""


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def _call_groq_eval(messages: list[dict]) -> str:
    """Call Groq with retry-backoff for rate limiting."""
    client = AsyncGroq(api_key=settings.GROQ_API_KEY)
    completion = await client.chat.completions.create(  # type: ignore[call-overload]
        model=settings.GROQ_EVAL_MODEL,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=settings.GROQ_EVAL_MAX_TOKENS,
        temperature=settings.GROQ_EVAL_TEMPERATURE,
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content or ""


def _parse_evaluation(raw: str) -> dict:
    """Parse the LLM evaluation response, handling markdown code fences."""
    # Strip markdown code fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse evaluation JSON. Raw output: {raw}")
        raise e

    # Validate and clamp score
    score = result.get("score", 5)
    if not isinstance(score, int | float):
        score = 5
    result["score"] = max(0, min(10, int(score)))

    # Ensure quality is valid
    if result.get("quality") not in ("strong", "adequate", "weak", "non_answer"):
        result["quality"] = "adequate"

    return result


def _is_technical_section(skill: str, section_name: str) -> bool:
    """Determine if the current section/skill is technical."""
    non_tech_indicators = {
        "communication",
        "behavioural",
        "behavioral",
        "closing",
        "general",
        "cultural",
        "culture",
        "self introduction",
        "intro",
        "introduction",
    }
    skill_lower = skill.lower()
    name_lower = section_name.lower()

    return not any(
        indicator in skill_lower or indicator in name_lower
        for indicator in non_tech_indicators
    )


async def evaluate_answer_node(state: InterviewState) -> dict:
    """
    Evaluate the candidate's answer using a structured LLM call.

    Updates consecutive counters, used_concepts, and appends to
    answer_evaluations. Then proceeds to generate_question.
    """
    transcript = state.get("last_transcript", "")
    question = state.get("current_question_text", "")
    sections = state.get("sections", [])
    section_idx = state.get("current_section_index", 0)
    used_concepts = list(state.get("used_concepts", []))
    turn_number = state.get("turn_number", 0)

    current_section = sections[section_idx] if section_idx < len(sections) else None
    skill = current_section["skill"] if current_section else "general"
    priority = current_section["priority_score"] if current_section else 5
    section_name = current_section["name"] if current_section else "unknown"
    difficulty = state.get("current_question_difficulty", "medium")

    if _is_technical_section(skill, section_name):
        difficulty_context = f"- Question difficulty: {difficulty}"
        scoring_instructions = """For EASY questions (maximum possible score is capped at 6):
- 1-2: Weak — shows superficial understanding, misses key points
- 3-4: Adequate — demonstrates basic understanding, covers main points
- 5-6: Strong/Exceptional — shows depth, provides examples (capped at 6)

For MEDIUM questions (maximum possible score is capped at 8):
- 1-3: Weak — shows superficial understanding, misses key points
- 4-6: Adequate — demonstrates basic understanding, covers main points
- 7-8: Strong/Exceptional — shows depth, provides examples, explains trade-offs (capped at 8)

For HARD questions (full score range up to 10):
- 1-3: Weak — shows superficial understanding, misses key points
- 4-6: Adequate — demonstrates basic understanding, covers main points
- 7-8: Strong — shows depth, provides examples, explains trade-offs
- 9-10: Exceptional — demonstrates mastery, novel insights, production experience"""
    else:
        difficulty_context = ""
        scoring_instructions = """Standard scoring guide (full score range up to 10):
- 1-3: Weak — shows superficial understanding or misses key conversational/behavioural points
- 4-6: Adequate — demonstrates basic competency, answers the prompt clearly
- 7-8: Strong — shows depth, detailed examples, clear communication
- 9-10: Exceptional — outstanding alignment, structure, and communication"""

    # Build the evaluation prompt
    prompt = _EVALUATION_PROMPT.format(
        skill=skill,
        priority=priority,
        question_text=question,
        difficulty_context=difficulty_context,
        transcript=transcript,
        used_concepts=", ".join(used_concepts) if used_concepts else "none yet",
        scoring_instructions=scoring_instructions,
    )

    raw_response = await _call_groq_eval(
        [
            {
                "role": "system",
                "content": "You are a precise evaluation engine. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]
    )
    evaluation = _parse_evaluation(raw_response)

    logger.info(
        "Answer evaluated: quality=%s score=%d turn=%d",
        evaluation["quality"],
        evaluation["score"],
        turn_number,
    )

    # Update consecutive counters
    consecutive_strong = state.get("consecutive_strong", 0)
    consecutive_weak = state.get("consecutive_weak", 0)

    if evaluation["quality"] == "strong":
        consecutive_strong += 1
        consecutive_weak = 0
    elif evaluation["quality"] in ("weak", "non_answer"):
        consecutive_weak += 1
        consecutive_strong = 0
    else:
        # adequate resets both
        consecutive_strong = 0
        consecutive_weak = 0

    # Update used_concepts
    key_concept = evaluation.get("key_concept_demonstrated", "")
    if key_concept and key_concept not in used_concepts:
        used_concepts.append(key_concept)

    # Build evaluation record
    eval_record = {
        "turn_number": turn_number,
        "section": section_name,
        "skill": skill,
        "question": question,
        "answer": transcript,
        **evaluation,
    }

    # Build transcript turn record for the candidate's answer
    turn_record = {
        "turn_number": turn_number,
        "speaker": "candidate",
        "text": transcript,
        "section": section_name,
        "type": "answer",
        "evaluation": evaluation,
    }

    return {
        "last_evaluation": evaluation,
        "consecutive_strong": consecutive_strong,
        "consecutive_weak": consecutive_weak,
        "used_concepts": used_concepts,
        "answer_evaluations": state.get("answer_evaluations", []) + [eval_record],
        "transcript_turns": state.get("transcript_turns", []) + [turn_record],
    }
