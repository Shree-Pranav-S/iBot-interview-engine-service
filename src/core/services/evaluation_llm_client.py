"""NVIDIA NIM client for one-shot holistic interview evaluation."""

from __future__ import annotations

import copy
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

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
from src.core.services.evaluation_context_builder import (
    EvaluationContextBundle,
)
from src.core.services.evaluation_errors import (
    EvaluationSchemaError,
    PermanentEvaluationError,
    TransientEvaluationError,
)
from src.core.services.evaluation_prompt import (
    HOLISTIC_EVALUATION_SYSTEM_PROMPT,
)
from src.schemas.evaluation_llm import HolisticEvaluationLLMOutput

logger = logging.getLogger(__name__)
_client: AsyncOpenAI | None = None


@dataclass(frozen=True)
class NvidiaEvaluationResult:
    output: HolisticEvaluationLLMOutput
    raw_output: dict[str, Any]


def _get_client() -> AsyncOpenAI:
    """
    Initialize and return a singleton AsyncOpenAI client configured for NVIDIA NIM.

    Returns:
        An instantiated AsyncOpenAI client.

    Raises:
        PermanentEvaluationError: If the NVIDIA NIM API key is missing.
    """
    global _client
    api_key = settings.NVIDIA_NIM_API_KEY.strip()
    if not api_key:
        raise PermanentEvaluationError(
            "NVIDIA_NIM_API_KEY is required for holistic evaluation"
        )
    if _client is None:
        _client = AsyncOpenAI(
            base_url=settings.NVIDIA_NIM_BASE_URL,
            api_key=api_key,
            timeout=settings.NVIDIA_NIM_TIMEOUT_SECS,
            # Provider failures are retried by Celery with a meaningful delay.
            # An immediate SDK retry after a five-minute 504 only causes a 429.
            max_retries=0,
        )
    return _client


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
    return errors


def _normalized_skill_key(value: Any) -> str:
    """
    Normalize a skill name into a consistent lowercase string for reliable matching.

    Args:
        value: The raw skill name.

    Returns:
        A normalized string containing only alphanumeric characters and allowed symbols (+, #, .).
    """
    return " ".join(re.findall(r"[a-z0-9+#.]+", str(value).casefold()))


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
    by_normalized_key = {
        _normalized_skill_key(key): item for key, item in value.items()
    }
    remapped: dict[str, Any] = {}
    for spec in bundle.technical_skills:
        if spec.name in value:
            remapped[spec.name] = value[spec.name]
            continue
        matching_value = by_normalized_key.get(_normalized_skill_key(spec.name))
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
    Construct the base chat messages to prompt the LLM for evaluation.

    Args:
        bundle: The full evaluation context bundle.

    Returns:
        A list of chat completion messages containing the system instructions and user input.
    """
    input_json = json.dumps(
        bundle.evaluation_input.model_dump(mode="json"),
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


async def _complete(
    messages: list[ChatCompletionMessageParam],
    *,
    enable_reasoning: bool = True,
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
    try:
        extra_body: dict[str, Any] = {
            "chat_template_kwargs": {
                "enable_thinking": enable_reasoning,
            },
        }
        if enable_reasoning:
            extra_body["reasoning_budget"] = settings.NVIDIA_NIM_REASONING_BUDGET
        if settings.NVIDIA_NIM_STREAM:
            stream_response = await _get_client().chat.completions.create(
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
                    content = getattr(delta, "content", None)
                    if content:
                        content_parts.append(str(content))
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
        else:
            completion_response = await _get_client().chat.completions.create(
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
                raise TransientEvaluationError(
                    "NVIDIA NIM returned no evaluation choices"
                )
            content = completion_response.choices[0].message.content
            elapsed = time.monotonic() - t0
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
    except RateLimitError as exc:
        elapsed = time.monotonic() - t0
        retry_after = _retry_after_seconds(exc)
        logger.error(
            "NVIDIA NIM rate limited after %.1f seconds: %s",
            elapsed,
            exc,
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


async def run_nvidia_holistic_evaluation(
    bundle: EvaluationContextBundle,
) -> NvidiaEvaluationResult:
    """
    Run one evaluation and, only if needed, one lightweight JSON repair.

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
        return result
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
        return _validate_output(repaired_raw, bundle)
    except EvaluationSchemaError as exc:
        raise EvaluationSchemaError(
            f"NVIDIA output remained invalid after retry and repair: {exc}"
        ) from exc
