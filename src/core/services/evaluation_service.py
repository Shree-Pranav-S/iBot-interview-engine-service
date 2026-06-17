import json
import logging
import uuid

from groq import AsyncGroq

from src.config.settings import settings
from src.data.repositories.interview_repository import (
    get_interview_data_for_evaluation,
    save_holistic_evaluation,
)

logger = logging.getLogger(__name__)

_HOLISTIC_SYSTEM_PROMPT = """You are a Principal Technical Recruiter and Hiring Manager.
Your task is to generate a final holistic interview evaluation report based on the provided interview transcript and turn-by-turn evaluations.

You must return your evaluation strictly as a valid JSON object matching the exact schema provided.

The JSON schema you must output:
{
  "skill_scores": {
    "skill_name": {
      "priority_score": int (1-10),
      "raw_score": int (0-10),
      "weighted_score": float,
      "summary": "string"
    }
  },
  "technical_dimension_score": float (0-100),
  "problem_solving_score": float (0-100),
  "problem_solving_evidence": ["string"],
  "problem_solving_summary": "string",
  "communication_score": float (0-100),
  "communication_evidence": ["string"],
  "communication_summary": "string",
  "behavioural_score": float (0-100),
  "behavioural_evidence": ["string"],
  "behavioural_summary": "string",
  "cultural_fit_score": float (0-100),
  "cultural_fit_evidence": ["string"],
  "cultural_fit_summary": "string",
  "section_summaries": {
    "section_name": {
      "summary": "string",
      "avg_score": float (0-10)
    }
  },
  "overall_score": float (0-100),
  "hiring_recommendation": "STRONG_HIRE" | "HIRE" | "CONSIDER" | "WEAK" | "NO_HIRE",
  "overall_narrative": "string (3-4 sentences)",
  "strengths": ["string"],
  "concerns": ["string"],
  "red_flags": [{"description": "string", "severity": "minor"|"critical"}],
  "violation_summary": null,
  "best_answer": {
    "question": "string",
    "turn_number": int,
    "section": "string",
    "reason": "string"
  },
  "weakest_answer": {
    "question": "string",
    "turn_number": int,
    "section": "string",
    "reason": "string"
  },
  "recommendation_reasoning": "string"
}

Ensure your evaluation is entirely unbiased, based heavily on the specific signals demonstrated in the transcript, and accurately reflects the provided turn-by-turn scores. Calculate `technical_dimension_score` as the priority-weighted average of `skill_scores`. Calculate `overall_score` as roughly: technical 40%, problem_solving 20%, communication 15%, behavioural 15%, cultural_fit 10%.
"""


async def run_holistic_evaluation(candidate_assessment_id: str) -> None:
    """
    Fetch interview data, generate a holistic evaluation via Groq,
    and persist the results to the database.
    """
    try:
        ca_id = uuid.UUID(candidate_assessment_id)
        data = await get_interview_data_for_evaluation(ca_id)
        if not data or not data["session"]:
            logger.error(
                f"Cannot run holistic eval: No data found for {candidate_assessment_id}"
            )
            return

        session = data["session"]
        turns = data["turns"]
        evals = data["evaluations"]

        if not turns:
            logger.warning(
                f"No transcript turns for {candidate_assessment_id}. Skipping eval."
            )
            return

        # Format input for LLM
        transcript_text = "\n".join(
            f"Turn {t.turn_number} [{t.speaker.upper()} - {t.section}]: {t.text}"
            for t in turns
        )

        evals_text = "\n".join(
            f"Turn {e.turn_number} [Skill: {e.skill}]: Score={e.score}/10, Quality={e.quality}\n"
            f"Feedback: {e.one_line_feedback}\n"
            f"Signals Present: {e.signals_present}, Missing: {e.signals_missing}"
            for e in evals
        )

        user_prompt = f"""
Please generate the comprehensive JSON evaluation for the following interview.

=== TRANSCRIPT ===
{transcript_text}

=== PER-TURN EVALUATIONS ===
{evals_text}
"""

        client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        completion = await client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": _HOLISTIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        reply_str = completion.choices[0].message.content or "{}"
        eval_json = json.loads(reply_str)

        # Merge violation summary if any
        eval_json["violation_summary"] = None

        await save_holistic_evaluation(ca_id, session.id, eval_json)

    except Exception:
        logger.exception(
            f"Failed to generate holistic evaluation for {candidate_assessment_id}"
        )
