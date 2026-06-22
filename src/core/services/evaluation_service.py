import json
import logging
import uuid

from src.core.services.llm_service import evaluate
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
    Runs the final holistic evaluation based on transcript and turn-by-turn evaluations.
    """
    try:
        ca_uuid = uuid.UUID(candidate_assessment_id)
    except ValueError:
        logger.error(f"Invalid UUID for evaluation: {candidate_assessment_id}")
        return

    data = await get_interview_data_for_evaluation(ca_uuid)
    if not data or not data["session"]:
        logger.warning(f"No interview data found for {candidate_assessment_id}")
        return

    session = data["session"]
    turns = data["turns"]
    evals = data["evaluations"]

    if not turns:
        logger.warning(f"No transcript turns found for {candidate_assessment_id}")
        return

    # Compile the transcript for the LLM
    transcript_text = ""
    for turn in turns:
        speaker = "Interviewer" if turn.is_interviewer else "Candidate"
        transcript_text += f"[{speaker} Turn {turn.turn_number} - Section: {turn.section_name}]: {turn.content}\n"

    # Compile the turn-by-turn evaluations
    evals_text = ""
    for ev in evals:
        evals_text += f"[Turn {ev.turn_number} Eval]: Score {ev.score}/10, Required signals hit: {ev.signals_demonstrated}. Missing: {ev.signals_missing}. Feedback: {ev.feedback_for_candidate}\n"

    violations_text = ""
    if session.violations:
        violations_text = f"Violations: {json.dumps(session.violations, indent=2)}\n"

    user_message = (
        f"Transcript:\n{transcript_text}\n"
        f"Turn-by-turn evaluations:\n{evals_text}\n"
        f"{violations_text}"
    )

    messages = [
        {"role": "system", "content": _HOLISTIC_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    try:
        response_json = await evaluate(messages)
        eval_data = json.loads(response_json)
    except Exception:
        logger.exception(
            f"Failed to generate or parse holistic evaluation for {ca_uuid}"
        )
        return

    # Save to database
    await save_holistic_evaluation(ca_uuid, session.id, eval_data)
    logger.info(f"Holistic evaluation completed and saved for {ca_uuid}")
