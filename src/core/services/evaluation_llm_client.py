"""NVIDIA NIM client for single-stage holistic evaluation."""

from __future__ import annotations

import json
import logging

from openai.types.chat import ChatCompletionMessageParam

from src.core.exceptions.evaluation import EvaluationSchemaError
from src.schemas.evaluation_llm import (
    HolisticEvaluationLLMOutput,
    NvidiaEvaluationResult,
)
from src.schemas.internal_evaluation import EvaluationContextBundle
from src.utils.evaluation_llm import _base_messages, _complete, _validate_output

logger = logging.getLogger(__name__)


async def run_nvidia_holistic_evaluation(
    bundle: EvaluationContextBundle,
) -> NvidiaEvaluationResult:
    """
    Run one single-stage evaluation and, only if needed, one lightweight JSON repair.

    Args:
        bundle: The fully built and validated evaluation context bundle.

    Returns:
        The validated NVIDIA evaluation result.

    Raises:
        EvaluationSchemaError: If the model fails to produce a valid response even after a repair attempt.
    """

    candidate_id = bundle.evaluation_input.candidate.candidate_assessment_id
    logger.info(
        "Starting NVIDIA holistic evaluation",
        extra={
            "candidate_assessment_id": candidate_id,
            "num_technical_skills": len(bundle.technical_skills),
            "num_violations": len(bundle.evaluation_input.violations),
        },
    )

    messages = _base_messages(bundle)
    logger.info(
        "NVIDIA evaluation LLM attempt 1/1",
        extra={"candidate_assessment_id": candidate_id},
    )
    last_raw = await _complete(messages)
    try:
        result = _validate_output(last_raw, bundle)
        logger.info(
            "NVIDIA evaluation validated successfully",
            extra={"candidate_assessment_id": candidate_id},
        )
        return NvidiaEvaluationResult(
            output=result.output,
            raw_output={
                "pipeline": "direct_qa",
                "evaluation": result.raw_output,
            },
        )
    except EvaluationSchemaError as exc:
        last_error = str(exc)
        logger.warning(
            "NVIDIA holistic output needs lightweight repair: %s",
            last_error,
            extra={"candidate_assessment_id": candidate_id},
        )

    expected_skills = {
        spec.name: {
            "questions_evaluated": spec.questions_asked,
            "priority_score": spec.priority_score,
        }
        for spec in bundle.technical_skills
    }
    expected_question_ids = [
        item.question_id for item in bundle.evaluation_input.qa_pairs
    ]
    schema_json = json.dumps(
        HolisticEvaluationLLMOutput.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    repair_messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You repair an existing interview-evaluation JSON object. "
                "Return exactly one valid JSON object, with no Markdown. "
                "Preserve all evidence and judgments unless a validation error "
                "requires a structural correction. Do not reevaluate the "
                "interview and do not expose reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                f"INVALID_JSON:\n{last_raw}\n\n"
                f"VALIDATION_ERRORS:\n{last_error}\n\n"
                "AUTHORITATIVE_SKILL_METADATA:\n"
                f"{json.dumps(expected_skills, separators=(',', ':'))}\n\n"
                "AUTHORITATIVE_QUESTION_IDS:\n"
                f"{json.dumps(expected_question_ids, separators=(',', ':'))}\n\n"
                f"REQUIRED_OUTPUT_SCHEMA:\n{schema_json}\n\n"
                "Repair the JSON structure only."
            ),
        },
    ]
    repaired_raw = await _complete(
        repair_messages,
        enable_reasoning=False,
    )
    try:
        repaired = _validate_output(repaired_raw, bundle)
        return NvidiaEvaluationResult(
            output=repaired.output,
            raw_output={
                "pipeline": "direct_qa",
                "evaluation": repaired.raw_output,
                "evaluation_repaired": True,
            },
        )
    except EvaluationSchemaError as exc:
        raise EvaluationSchemaError(
            f"NVIDIA output remained invalid after retry and repair: {exc}"
        ) from exc
