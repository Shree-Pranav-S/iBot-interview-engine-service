"""Private helpers for NVIDIA holistic evaluation LLM calls."""

from __future__ import annotations

import copy
import json
import logging
import time
from typing import Any

import httpx
from langsmith import traceable
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

from src.config.settings import settings
from src.core.exceptions.evaluation import (
    EvaluationSchemaError,
    PermanentEvaluationError,
    TransientEvaluationError,
)
from src.core.services.evaluation_prompt import (
    HOLISTIC_EVALUATION_SYSTEM_PROMPT,
)
from src.schemas.evaluation_llm import (
    HolisticEvaluationLLMOutput,
    NvidiaEvaluationResult,
)
from src.schemas.internal_evaluation import EvaluationContextBundle
from src.utils.evaluation_context import _normalize_skill_key

logger = logging.getLogger(__name__)
_client: AsyncOpenAI | None = None
_fallback_client: AsyncOpenAI | None = None


def _get_client(use_fallback: bool = False) -> AsyncOpenAI:
    """
    Initialize and return a singleton AsyncOpenAI client configured for NVIDIA NIM.

    Returns:
        An instantiated AsyncOpenAI client.

    Raises:
        PermanentEvaluationError: If the NVIDIA NIM API key is missing.
    """
    global _client, _fallback_client
    api_key = (
        settings.FALLBACK_NVIDIA_NIM_API_KEY
        if use_fallback
        else settings.NVIDIA_NIM_API_KEY
    ).strip()
    if not api_key:
        message = (
            "FALLBACK_NVIDIA_NIM_API_KEY is required for fallback holistic evaluation"
            if use_fallback
            else "NVIDIA_NIM_API_KEY is required for holistic evaluation"
        )
        raise PermanentEvaluationError(message)
    client = _fallback_client if use_fallback else _client
    if client is None:
        client = AsyncOpenAI(
            base_url=settings.NVIDIA_NIM_BASE_URL,
            api_key=api_key,
            timeout=settings.NVIDIA_NIM_TIMEOUT_SECS,
            # Provider failures are retried by Celery with a meaningful delay.
            # An immediate SDK retry after a five-minute 504 only causes a 429.
            max_retries=0,
        )
        if use_fallback:
            _fallback_client = client
        else:
            _client = client
    return client


def _extract_json_object(raw: str) -> dict[str, Any]:
    """
    Safely extract and parse a JSON object from a raw LLM response.
    It handles common model artifacts like markdown code blocks.

    Args:
        raw: The raw text response from the LLM.

    Returns:
        A parsed JSON dictionary.

    Raises:
        EvaluationSchemaError: If no valid JSON object can be extracted or parsed.
    """
    value = str(raw or "").strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        if start < 0 or end <= start:
            raise EvaluationSchemaError(
                "NVIDIA evaluator response did not contain a JSON object"
            )
        try:
            parsed = json.loads(value[start : end + 1])
        except json.JSONDecodeError as exc:
            raise EvaluationSchemaError(
                f"NVIDIA evaluator returned invalid JSON: {exc}"
            ) from exc
    if not isinstance(parsed, dict):
        raise EvaluationSchemaError("NVIDIA evaluator response must be one JSON object")
    return parsed


def _semantic_errors(
    output: HolisticEvaluationLLMOutput,
    bundle: EvaluationContextBundle,
) -> list[str]:
    """
    Perform deep semantic validation on the parsed LLM output against the ground truth constraints.

    Args:
        output: The strongly-typed LLM output.
        bundle: The immutable evaluation context bundle containing expected values.

    Returns:
        A list of validation error strings. Returns an empty list if fully valid.
    """
    expected = {item.name for item in bundle.technical_skills}
    actual = set(output.skill_scores)
    errors: list[str] = []
    if actual != expected:
        errors.append(
            "skill keys must exactly equal planned technical skills: "
            f"{sorted(expected)}"
        )
    if output.violation_summary.validated_violation_count > len(
        bundle.evaluation_input.violations
    ):
        errors.append("validated_violation_count cannot exceed supplied violations")
    expected_question_ids = [
        item.question_id for item in bundle.evaluation_input.qa_pairs
    ]
    actual_question_ids = [item.question_id for item in output.question_evaluations]
    if actual_question_ids != expected_question_ids:
        errors.append(
            "question_evaluations must contain each authoritative question_id "
            f"exactly once and in input order: {expected_question_ids}"
        )
    return errors


def _remap_skill_dict(
    value: Any,
    bundle: EvaluationContextBundle,
) -> dict[str, Any]:
    """
    Attempt to correct slight deviations in skill names outputted by the LLM by matching
    them back to the exact planned skill names in the evaluation context.

    Args:
        value: The dictionary mapping skill names to some details.
        bundle: The authoritative evaluation context containing correct skill names.

    Returns:
        A new dictionary with corrected skill keys where possible.
    """
    if not isinstance(value, dict):
        return {}
    by_normalized_key = {_normalize_skill_key(key): item for key, item in value.items()}
    remapped: dict[str, Any] = {}
    for spec in bundle.technical_skills:
        if spec.name in value:
            remapped[spec.name] = value[spec.name]
            continue
        matching_value = by_normalized_key.get(_normalize_skill_key(spec.name))
        if matching_value is not None:
            remapped[spec.name] = matching_value
    return remapped


def _normalize_structural_output(
    parsed: dict[str, Any],
    bundle: EvaluationContextBundle,
) -> tuple[dict[str, Any], list[str]]:
    """
    Correct non-judgmental model bookkeeping before strict validation.

    This fixes trivial formatting errors or missed exact values (like question counts)
    that do not affect the qualitative assessment, reducing the need for costly LLM retries.

    Args:
        parsed: The parsed JSON dictionary from the LLM.
        bundle: The authoritative evaluation context.

    Returns:
        A tuple containing the corrected dictionary and a list of adjustment descriptions applied.
    """

    normalized = copy.deepcopy(parsed)
    adjustments: list[str] = []

    score_map = _remap_skill_dict(
        normalized.get("skill_scores"),
        bundle,
    )
    summary_map = _remap_skill_dict(
        normalized.get("skill_summary"),
        bundle,
    )
    evidence_map = _remap_skill_dict(
        normalized.get("skill_evidence"),
        bundle,
    )

    for spec in bundle.technical_skills:
        details = score_map.get(spec.name)
        if isinstance(details, dict):
            details = dict(details)
            if details.get("questions_evaluated") != spec.questions_asked:
                adjustments.append(
                    f"{spec.name}.questions_evaluated={spec.questions_asked}"
                )
            if details.get("priority_score") != spec.priority_score:
                adjustments.append(f"{spec.name}.priority_score={spec.priority_score}")
            details["questions_evaluated"] = spec.questions_asked
            details["priority_score"] = spec.priority_score
            score_map[spec.name] = details
        elif spec.questions_asked == 0:
            score_map[spec.name] = {
                "score": 0.0,
                "priority_score": spec.priority_score,
                "questions_evaluated": 0,
                "confidence": 0.0,
            }
            adjustments.append(f"added unassessed score entry for {spec.name}")

        if spec.questions_asked == 0:
            if not str(summary_map.get(spec.name) or "").strip():
                summary_map[spec.name] = (
                    "This planned skill was not assessed because no scored "
                    "question was asked."
                )
                adjustments.append(f"added unassessed summary for {spec.name}")
            if spec.name not in evidence_map:
                evidence_map[spec.name] = []

    if score_map:
        normalized["skill_scores"] = score_map
    if summary_map:
        normalized["skill_summary"] = summary_map
    if evidence_map or "skill_evidence" in normalized:
        normalized["skill_evidence"] = evidence_map

    authoritative_questions = {
        item.question_id: item for item in bundle.evaluation_input.qa_pairs
    }
    question_evaluations = normalized.get("question_evaluations")
    if isinstance(question_evaluations, list):
        for index, value in enumerate(question_evaluations):
            if not isinstance(value, dict):
                continue
            question = authoritative_questions.get(str(value.get("question_id") or ""))
            if question is None:
                continue
            for field in (
                "question_id",
                "section",
                "skill",
                "difficulty",
                "question_text",
                "answered",
            ):
                authoritative_value = getattr(question, field)
                if value.get(field) != authoritative_value:
                    adjustments.append(
                        f"question_evaluations[{index}].{field}=authoritative"
                    )
                value[field] = authoritative_value

    violation_summary = normalized.get("violation_summary")
    if isinstance(violation_summary, dict):
        violation_summary = dict(violation_summary)
        severity_counts = violation_summary.get("severity_counts")
        if isinstance(severity_counts, dict):
            try:
                count = sum(
                    max(0, int(severity_counts.get(level, 0)))
                    for level in ("low", "medium", "high", "critical")
                )
            except (TypeError, ValueError):
                count = None
            if count is not None:
                if violation_summary.get("validated_violation_count") != count:
                    adjustments.append(f"validated_violation_count={count}")
                violation_summary["validated_violation_count"] = count
                violation_summary["has_violation"] = count > 0
                normalized["violation_summary"] = violation_summary

    return normalized, adjustments


def _validate_output(
    raw: str,
    bundle: EvaluationContextBundle,
) -> NvidiaEvaluationResult:
    """
    Parse, normalize, and strictly validate the raw LLM output text against the context bundle.

    Args:
        raw: The raw text response from the LLM.
        bundle: The authoritative evaluation context bundle.

    Returns:
        A validated NVIDIA evaluation result and its raw dictionary.

    Raises:
        EvaluationSchemaError: If the output fails structural or semantic validation.
    """
    parsed = _extract_json_object(raw)
    normalized, adjustments = _normalize_structural_output(parsed, bundle)
    if adjustments:
        logger.warning(
            "Normalized minor NVIDIA evaluator output inconsistencies: %s",
            "; ".join(adjustments),
            extra={
                "candidate_assessment_id": (
                    bundle.evaluation_input.candidate.candidate_assessment_id
                ),
                "normalization_count": len(adjustments),
            },
        )
    try:
        output = HolisticEvaluationLLMOutput.model_validate(normalized)
    except ValidationError as exc:
        raise EvaluationSchemaError(str(exc)) from exc
    semantic_errors = _semantic_errors(output, bundle)
    if semantic_errors:
        raise EvaluationSchemaError("; ".join(semantic_errors))
    return NvidiaEvaluationResult(output=output, raw_output=parsed)


def _base_messages(
    bundle: EvaluationContextBundle,
) -> list[ChatCompletionMessageParam]:
    """
    Construct the chat messages to prompt the LLM for single-stage evaluation.

    Args:
        bundle: The full evaluation context bundle.

    Returns:
        A list of chat completion messages containing the system instructions and user input.
    """
    evaluation_input = bundle.evaluation_input.model_dump(mode="json")
    # Raw turns remain persisted and hashed, but the scorer receives only the
    # deterministic Q&A representation.
    evaluation_input.pop("transcript", None)
    input_json = json.dumps(
        evaluation_input,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    schema_json = json.dumps(
        HolisticEvaluationLLMOutput.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return [
        {
            "role": "system",
            "content": HOLISTIC_EVALUATION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": (
                "EVALUATION_INPUT_JSON:\n"
                f"{input_json}\n\n"
                "REQUIRED_OUTPUT_SCHEMA:\n"
                f"{schema_json}\n\n"
                "Evaluate the complete interview now. Return strict JSON only."
            ),
        },
    ]


def _retry_after_seconds(exc: APIStatusError) -> int | None:
    """
    Extract the 'retry-after' header from an API error response.

    Args:
        exc: The APIStatusError exception raised by the OpenAI client.

    Returns:
        The recommended wait time in seconds (clamped to 900s), or None if unavailable.
    """
    value = exc.response.headers.get("retry-after")
    if not value:
        return None
    try:
        return max(1, min(900, int(float(value))))
    except (TypeError, ValueError):
        return None


async def _complete_non_stream(
    messages: list[ChatCompletionMessageParam],
    *,
    extra_body: dict[str, Any],
    use_fallback: bool,
    t0: float,
) -> tuple[str, float]:
    """Execute one non-streaming NVIDIA NIM completion."""
    completion_response = await _get_client(use_fallback).chat.completions.create(
        model=settings.NVIDIA_NIM_MODEL,
        messages=messages,
        temperature=settings.NVIDIA_NIM_TEMPERATURE,
        top_p=settings.NVIDIA_NIM_TOP_P,
        max_tokens=settings.NVIDIA_NIM_MAX_TOKENS,
        response_format={"type": "json_object"},
        extra_body=extra_body,
        stream=False,
    )
    if not completion_response.choices:
        raise TransientEvaluationError("NVIDIA NIM returned no evaluation choices")
    content = completion_response.choices[0].message.content
    return str(content or ""), time.monotonic() - t0


@traceable(name="nvidia_holistic_evaluation", run_type="llm")
async def _complete(
    messages: list[ChatCompletionMessageParam],
    *,
    enable_reasoning: bool = True,
    use_fallback: bool = False,
) -> str:
    """
    Execute a chat completion call to the NVIDIA NIM LLM.

    Args:
        messages: The chat messages.
        enable_reasoning: Whether Nemotron should emit internal reasoning tokens.

    Returns:
        The full string response content.

    Raises:
        TransientEvaluationError: For retryable API errors or timeouts.
        PermanentEvaluationError: For definitive API rejections (e.g., 400 Bad Request).
    """
    message_chars = sum(len(str(item.get("content") or "")) for item in messages)
    logger.info(
        "NVIDIA NIM LLM call starting",
        extra={
            "model": settings.NVIDIA_NIM_MODEL,
            "reasoning_enabled": enable_reasoning,
            "timeout_secs": settings.NVIDIA_NIM_TIMEOUT_SECS,
            "max_tokens": settings.NVIDIA_NIM_MAX_TOKENS,
            "stream": settings.NVIDIA_NIM_STREAM,
            "message_chars": message_chars,
        },
    )
    t0 = time.monotonic()
    content = ""
    elapsed = 0.0
    try:
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {
                "enable_thinking": enable_reasoning,
            },
        }
        if enable_reasoning:
            extra_body["reasoning_budget"] = settings.NVIDIA_NIM_REASONING_BUDGET
        if settings.NVIDIA_NIM_STREAM:
            try:
                stream_response = await _get_client(
                    use_fallback
                ).chat.completions.create(
                    model=settings.NVIDIA_NIM_MODEL,
                    messages=messages,
                    temperature=settings.NVIDIA_NIM_TEMPERATURE,
                    top_p=settings.NVIDIA_NIM_TOP_P,
                    max_tokens=settings.NVIDIA_NIM_MAX_TOKENS,
                    response_format={"type": "json_object"},
                    extra_body=extra_body,
                    stream=True,
                )
                content_parts: list[str] = []
                reasoning_chars = 0
                chunk_count = 0
                first_chunk_secs: float | None = None
                async for chunk in stream_response:
                    chunk_count += 1
                    if first_chunk_secs is None:
                        first_chunk_secs = time.monotonic() - t0
                        logger.info(
                            "NVIDIA NIM stream opened after %.1f seconds",
                            first_chunk_secs,
                        )
                    for choice in chunk.choices:
                        delta = choice.delta
                        delta_content = getattr(delta, "content", None)
                        if delta_content:
                            content_parts.append(str(delta_content))
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning:
                            reasoning_chars += len(str(reasoning))
                content = "".join(content_parts)
                elapsed = time.monotonic() - t0
                logger.info(
                    "NVIDIA NIM stream completed in %.1f seconds",
                    elapsed,
                    extra={
                        "elapsed_secs": round(elapsed, 1),
                        "first_chunk_secs": (
                            round(first_chunk_secs, 1)
                            if first_chunk_secs is not None
                            else None
                        ),
                        "chunk_count": chunk_count,
                        "reasoning_chars": reasoning_chars,
                        "response_length": len(content),
                    },
                )
            except httpx.TransportError as exc:
                elapsed = time.monotonic() - t0
                logger.warning(
                    "NVIDIA NIM stream interrupted after %.1f seconds; "
                    "retrying with non-streaming completion: %s",
                    elapsed,
                    exc,
                )
                content, elapsed = await _complete_non_stream(
                    messages,
                    extra_body=extra_body,
                    use_fallback=use_fallback,
                    t0=time.monotonic(),
                )
        else:
            content, elapsed = await _complete_non_stream(
                messages,
                extra_body=extra_body,
                use_fallback=use_fallback,
                t0=t0,
            )
    except APITimeoutError as exc:
        elapsed = time.monotonic() - t0
        logger.error(
            "NVIDIA NIM request TIMED OUT after %.1f seconds (limit: %.0fs)",
            elapsed,
            settings.NVIDIA_NIM_TIMEOUT_SECS,
        )
        raise TransientEvaluationError(
            f"NVIDIA NIM timed out after {elapsed:.1f}s: {exc}",
            retry_after_seconds=60,
        ) from exc
    except APIConnectionError as exc:
        elapsed = time.monotonic() - t0
        logger.error(
            "NVIDIA NIM connection error after %.1f seconds: %s",
            elapsed,
            exc,
        )
        raise TransientEvaluationError(f"NVIDIA NIM connection error: {exc}") from exc
    except httpx.TransportError as exc:
        elapsed = time.monotonic() - t0
        logger.error(
            "NVIDIA NIM transport error after %.1f seconds: %s",
            elapsed,
            exc,
        )
        raise TransientEvaluationError(
            f"NVIDIA NIM transport error: {exc}",
            retry_after_seconds=60,
        ) from exc
    except RateLimitError as exc:
        elapsed = time.monotonic() - t0
        retry_after = _retry_after_seconds(exc)
        logger.error(
            "NVIDIA NIM rate limited after %.1f seconds: %s",
            elapsed,
            exc,
        )
        if not use_fallback and settings.FALLBACK_NVIDIA_NIM_API_KEY.strip():
            logger.info(
                "Retrying with fallback NVIDIA NIM API key due to RateLimitError..."
            )
            return await _complete(
                messages,
                enable_reasoning=enable_reasoning,
                use_fallback=True,
            )
        raise TransientEvaluationError(
            f"NVIDIA NIM rate limited: {exc}",
            retry_after_seconds=retry_after or 60,
        ) from exc
    except APIStatusError as exc:
        elapsed = time.monotonic() - t0
        logger.error(
            "NVIDIA NIM HTTP %d error after %.1f seconds: %s",
            exc.status_code,
            elapsed,
            exc,
        )
        if (
            exc.status_code == 429
            and not use_fallback
            and settings.FALLBACK_NVIDIA_NIM_API_KEY.strip()
        ):
            logger.info("Retrying with fallback NVIDIA NIM API key due to HTTP 429...")
            return await _complete(
                messages,
                enable_reasoning=enable_reasoning,
                use_fallback=True,
            )
        if exc.status_code in {408, 409, 425, 429} or exc.status_code >= 500:
            raise TransientEvaluationError(
                f"Temporary NVIDIA NIM HTTP {exc.status_code}",
                retry_after_seconds=(
                    _retry_after_seconds(exc)
                    or (60 if exc.status_code >= 500 else None)
                ),
            ) from exc
        raise PermanentEvaluationError(
            f"NVIDIA NIM rejected the request with HTTP {exc.status_code}"
        ) from exc
    except APIError as exc:
        elapsed = time.monotonic() - t0
        logger.error(
            "NVIDIA NIM generic API error after %.1f seconds: %s",
            elapsed,
            exc,
        )
        raise TransientEvaluationError(
            f"NVIDIA NIM API error: {exc}",
            retry_after_seconds=60,
        ) from exc

    if not content:
        logger.error("NVIDIA NIM returned empty content after %.1f seconds", elapsed)
        raise TransientEvaluationError(
            "NVIDIA NIM returned an empty evaluation response"
        )

    logger.info(
        "NVIDIA NIM LLM call completed in %.1f seconds",
        elapsed,
        extra={
            "elapsed_secs": round(elapsed, 1),
            "response_length": len(content),
        },
    )
    return content
