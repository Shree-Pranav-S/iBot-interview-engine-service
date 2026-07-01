"""Deterministic fast paths and strict LLM response classification."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from src.control.agents.key_routing import active_turn_key_slot
from src.control.agents.nodes.llm_helpers import classify_with_schema
from src.control.agents.prompts import CLASSIFICATION_SYSTEM_PROMPT
from src.control.agents.state import InterviewState
from src.schemas.prompts import CandidateResponseClassification
from src.utils.interview_graph import deterministic_violation_id, utc_now_iso

logger = logging.getLogger(__name__)

REPEAT_PATTERN = re.compile(
    r"\b("
    r"repeat (?:it|that|this question|that question|the question)|"
    r"can you repeat (?:it|that|the question|this question)|"
    r"could you repeat (?:it|that|the question|this question)|"
    r"say (?:it|that) again|hear (?:it|that|this question|the question) again|"
    r"once more"
    r")\b",
    re.IGNORECASE,
)
REPHRASE_PATTERN = re.compile(
    r"\b((?:can|could|would|will) you (?:please )?rephrase|please rephrase|"
    r"rephrase (?:it|that|this|the question)|"
    r"phrase (?:it|that) differently|put (?:it|that) another way|"
    r"simplify (?:it|that|the question)|make the question (?:simpler|clearer)|say (?:it|that) (?:differently|in simpler words)|(?:didn(?:'|')?t|did not) understand)\b",
    re.IGNORECASE,
)
SKIP_PATTERN = re.compile(
    r"\b(skip (?:it|this|that|the question)|"
    r"pass on (?:it|this|that|the question)|"
    r"move (?:on|to the next question)|next question|"
    r"i (?:do not|don't|dont) know(?: the answer)?|"
    r"i (?:cannot|can't|cant) answer|no idea)\b",
    re.IGNORECASE,
)
THINK_PATTERN = re.compile(
    r"\b(?:i\s+)?(?:need|want|would like|could use)\s+"
    r"(?:some|a|a few|more|a little|a bit of)?\s*"
    r"(?:time|a moment|seconds?|a minute)\s+(?:to\s+)?"
    r"(?:think|consider|prepare|gather my thoughts)\b|"
    r"\b(?:can|could|may)\s+i\s+(?:take|have|get)\s+"
    r"(?:a|some|a few)?\s*(?:moment|time|seconds?|minute)\s+"
    r"(?:to\s+)?(?:think|consider|prepare)\b|\b(?:hold on|give me a (?:moment|second)|let me think)\b|"
    r"\b(?:give me|can i have|could i have|may i have)\s+"
    r"(?:a|one|some)?\s*(?:moment|minute|few seconds)\b",
    re.IGNORECASE,
)
YES_PATTERN = re.compile(
    r"^\s*(yes|yeah|yep|sure|please|okay|ok|i do|that would help)\b",
    re.IGNORECASE,
)
NO_PATTERN = re.compile(
    r"^\s*(no|nope|not really|i am (?:fine|ready)|i'm (?:fine|ready))\b",
    re.IGNORECASE,
)
YEARS_OF_EXPERIENCE_PATTERN = re.compile(
    r"\b(?:i\s+(?:have|bring|possess))\s+"
    r"(?:about|around|roughly|nearly|over|more than)?\s*"
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\b"
    r"(?:\s+of\s+(?:professional\s+|work\s+|industry\s+)?experience)?|"
    r"\bwith\s+(?:about|around|roughly|nearly|over|more than)?\s*"
    r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+of\s+"
    r"(?:professional\s+|work\s+|industry\s+)?experience\b",
    re.IGNORECASE,
)
SELF_INTRO_ELABORATION_WORD_THRESHOLD = 15
BEHAVIOURAL_ANSWER_WORD_THRESHOLD = 15
BEHAVIOURAL_SUBSTANTIAL_WORD_THRESHOLD = 20
DETERMINISTIC_CLASSIFICATION_SOURCES = frozenset(
    {
        "deterministic",
        "deterministic_self_intro",
        "deterministic_behavioural",
    }
)
QUESTION_LIKE_ANSWER_PATTERN = re.compile(
    r"^\s*(?:when you say|what do you mean|do you mean|did you mean|"
    r"are you asking|should i|would you like|does that include|"
    r"is the question|can you clarify|could you clarify)\b",
    re.IGNORECASE,
)
IRRELEVANT_TOPIC_PATTERN = re.compile(
    r"\b(?:weather|lunch|dinner|weekend|football|cricket|movie|netflix|"
    r"boyfriend|girlfriend|married|kids?|politics|religion|salary|pay|"
    r"wifi|internet (?:is )?down|cannot hear|can(?:'|')?t hear|audio (?:issue|problem)|"
    r"tell me the answer|give me the answer|what is the correct answer)\b",
    re.IGNORECASE,
)

CONTENT_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[+#.-][a-z0-9]+)*")
CLASSIFICATION_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "also",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "describe",
        "did",
        "do",
        "does",
        "explain",
        "for",
        "from",
        "give",
        "how",
        "i",
        "in",
        "into",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "please",
        "tell",
        "that",
        "the",
        "their",
        "this",
        "to",
        "use",
        "using",
        "was",
        "what",
        "when",
        "which",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


def _spoken_word_count(text: str) -> int:
    """Count spoken words using the same technical-token rules as classification."""

    return len(re.findall(r"\b[\w+#.-]+\b", text))


def is_deterministic_classification_source(source: str | None) -> bool:
    """Return True when a turn was classified without the merged interviewer LLM."""

    return str(source or "") in DETERMINISTIC_CLASSIFICATION_SOURCES


def _is_developed_spoken_answer(text: str, *, min_words: int = 10) -> bool:
    """Detect longer answer attempts that should not match clarification regexes."""

    if _spoken_word_count(text) < min_words:
        return False
    return not (
        text.rstrip().endswith("?") or QUESTION_LIKE_ANSWER_PATTERN.search(text)
    )


def will_bypass_interviewer_llm(state: InterviewState, text: str) -> bool:
    """Mirror interviewer_turn pre-LLM routing for filler suppression."""

    det = _deterministic_classification(state, text)
    if det is not None and det.get("response_type") in (
        "silence",
        "irrelevant",
        "clarification",
    ):
        return True
    if bool(state.get("is_self_introduction")):
        return True
    word_count = _spoken_word_count(text)
    return (
        state.get("current_section_kind") == "behavioural_cultural"
        and word_count > BEHAVIOURAL_ANSWER_WORD_THRESHOLD
    )


def _content_tokens(text: str) -> set[str]:
    """Return meaningful lowercase tokens for conservative relevance matching."""

    return {
        token
        for token in CONTENT_TOKEN_PATTERN.findall(text.casefold())
        if len(token) > 1 and token not in CLASSIFICATION_STOP_WORDS
    }


def _is_clear_relevant_answer(state: InterviewState, text: str) -> bool:
    """
    Recognize only high-confidence answer attempts without an LLM round trip.

    Two meaningful terms shared with the active question are enough to establish
    relevance for a developed spoken response. Question-like utterances remain
    model-classified so genuine scope doubts are not mistaken for answers.
    """

    if _spoken_word_count(text) < 10:
        return False
    if text.rstrip().endswith("?") or QUESTION_LIKE_ANSWER_PATTERN.search(text):
        return False

    question_tokens = _content_tokens(str(state.get("current_question_text") or ""))
    response_tokens = _content_tokens(text)
    return len(question_tokens & response_tokens) >= 2


def _result(
    *,
    response_type: str,
    clarification_type: str | None,
    is_substantial: bool | None,
    question_doubt_response: str | None = None,
) -> dict[str, Any]:
    """
    Construct a standardized classification result dictionary.

    Args:
        response_type: The broad category of the response ('answer', 'clarification', 'silence', etc.).
        clarification_type: Specific sub-type if the response is a clarification request.
        is_substantial: Whether the response contains enough substance to evaluate.
        question_doubt_response: Optional generated response for doubt cases.

    Returns:
        The dictionary matching CandidateResponseClassification structure.
    """
    return {
        "response_type": response_type,
        "clarification_type": clarification_type,
        "is_substantial": is_substantial,
        "question_doubt_response": question_doubt_response,
    }


def _is_likely_irrelevant(state: InterviewState, text: str) -> bool:
    """Conservative off-topic detection before an LLM round trip."""

    if IRRELEVANT_TOPIC_PATTERN.search(text):
        return True
    word_count = _spoken_word_count(text)
    if word_count <= 2:
        return False
    question_tokens = _content_tokens(str(state.get("current_question_text") or ""))
    response_tokens = _content_tokens(text)
    if not question_tokens or not response_tokens:
        return False
    overlap = len(question_tokens & response_tokens)
    if word_count >= 8 and overlap == 0:
        return True
    return (
        word_count >= 12 and overlap <= 1 and not _is_clear_relevant_answer(state, text)
    )


def _deterministic_classification(
    state: InterviewState,
    text: str,
) -> dict[str, Any] | None:
    """
    Attempt to classify candidate speech using high-confidence regex patterns.
    This saves LLM calls for obvious cases like silence, 'yes/no', skips, and repeat requests.

    Args:
        state: The current interview state.
        text: The transcribed text from the candidate.

    Returns:
        A classification dictionary if a pattern matched, or None if the LLM is needed.
    """
    event = state.get("candidate_event") or {}
    if isinstance(event, dict) and event.get("event_type") == "silence_timeout":
        return _result(
            response_type="silence",
            clarification_type=None,
            is_substantial=None,
        )

    if state.get("silence_stage") == "awaiting_think_confirmation":
        if YES_PATTERN.search(text):
            return _result(
                response_type="clarification",
                clarification_type="time_to_think",
                is_substantial=None,
            )
        if NO_PATTERN.search(text):
            return _result(
                response_type="clarification",
                clarification_type="decline_think_time",
                is_substantial=None,
            )

    if SKIP_PATTERN.search(text) and not _is_developed_spoken_answer(text):
        return _result(
            response_type="clarification",
            clarification_type="skip_question",
            is_substantial=None,
        )
    if REPEAT_PATTERN.search(text):
        return _result(
            response_type="clarification",
            clarification_type="repeat_question",
            is_substantial=None,
        )
    if REPHRASE_PATTERN.search(text):
        return _result(
            response_type="clarification",
            clarification_type="rephrase_question",
            is_substantial=None,
        )
    if THINK_PATTERN.search(text):
        return _result(
            response_type="clarification",
            clarification_type="time_to_think",
            is_substantial=None,
        )

    if bool(state.get("is_self_introduction")):
        word_count = _spoken_word_count(text)
        return _result(
            response_type="answer",
            clarification_type=None,
            is_substantial=(word_count >= SELF_INTRO_ELABORATION_WORD_THRESHOLD),
        )
    if state.get("current_section_kind") == "behavioural_cultural":
        word_count = _spoken_word_count(text)
        if word_count > BEHAVIOURAL_ANSWER_WORD_THRESHOLD:
            return _result(
                response_type="answer",
                clarification_type=None,
                is_substantial=(word_count > BEHAVIOURAL_SUBSTANTIAL_WORD_THRESHOLD),
            )
    if _is_clear_relevant_answer(state, text):
        return _result(
            response_type="answer",
            clarification_type=None,
            is_substantial=True,
        )
    if _is_likely_irrelevant(state, text):
        return _result(
            response_type="irrelevant",
            clarification_type=None,
            is_substantial=None,
        )
    return None


def _classification_messages(state: InterviewState) -> list[dict[str, str]]:
    """
    Construct the messages to prompt the LLM to classify the candidate's response.

    Args:
        state: The current interview state.

    Returns:
        A list of chat messages for the LLM.
    """
    context = {
        "previous_candidate_response": state.get("previous_candidate_response") or "",
        "previous_question": state.get("current_question_text") or "",
        "is_self_introduction": bool(state.get("is_self_introduction")),
    }
    return [
        {"role": "system", "content": CLASSIFICATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(context, ensure_ascii=False, default=str),
        },
    ]


def _safe_fallback(text: str) -> CandidateResponseClassification:
    """
    Keep a live interview moving without accepting malformed model JSON.
    Used if the classification LLM call fails completely.

    Args:
        text: The raw transcribed text.

    Returns:
        A default safe classification treating the text as an answer.
    """

    return CandidateResponseClassification(
        response_type="answer",
        clarification_type=None,
        is_substantial=_spoken_word_count(text) > 2,
        question_doubt_response=None,
    )


def _resume_has_skill(state: InterviewState) -> bool:
    """
    Check if the currently active technical skill is explicitly listed on the candidate's resume.

    Args:
        state: The current interview state.

    Returns:
        True if there is a match, False otherwise.
    """
    skill = str(state.get("current_technical_skill") or "").strip().casefold()
    if not skill:
        return False
    for raw in (state.get("resume_context") or {}).get("skills", []):
        resume_skill = str(raw).strip().casefold()
        if resume_skill == skill:
            return True
        if (
            resume_skill
            and min(len(resume_skill), len(skill)) >= 3
            and (resume_skill in skill or skill in resume_skill)
        ):
            return True
    return False


def _violation(
    state: InterviewState,
    violation_type: str,
    *,
    severity: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Construct a deterministic violation record for proctoring.

    Args:
        state: The current interview state.
        violation_type: The specific code for the violation.
        severity: 'low', 'medium', or 'high'.
        metadata: Additional context for the violation.

    Returns:
        A structured violation dictionary.
    """
    return {
        "violation_id": deterministic_violation_id(
            str(state["interview_session_id"]),
            int(state.get("turn_number") or 1),
            violation_type,
        ),
        "turn_number": int(state.get("turn_number") or 1),
        "violation_type": violation_type,
        "candidate_transcript": str(state.get("previous_candidate_response") or ""),
        "severity": severity,
        "timestamp": utc_now_iso(),
        "metadata": metadata,
    }


def _turn_violations(
    state: InterviewState,
    *,
    response_type: str,
    clarification_type: str | None,
    resume_skill_match: bool,
) -> list[dict[str, Any]]:
    """
    Analyze the classification to detect and generate proctoring violations for this turn.
    Catches irrelevant answers, skipped resume skills, and experience inflation.

    Args:
        state: The current interview state.
        response_type: The determined response type.
        clarification_type: The determined clarification type, if any.
        resume_skill_match: Whether the current skill is on the resume.

    Returns:
        A list of triggered violation dictionaries.
    """
    violations: list[dict[str, Any]] = []
    if response_type == "irrelevant":
        violations.append(
            _violation(
                state,
                "irrelevant_response",
                severity="low",
                metadata={
                    "question_id": state.get("current_question_id"),
                    "current_skill": state.get("current_technical_skill"),
                },
            )
        )

    if clarification_type == "skip_question" and resume_skill_match:
        violations.append(
            _violation(
                state,
                "skipped_resume_skill",
                severity="medium",
                metadata={
                    "question_id": state.get("current_question_id"),
                    "current_skill": state.get("current_technical_skill"),
                },
            )
        )

    if response_type == "answer":
        match = YEARS_OF_EXPERIENCE_PATTERN.search(
            str(state.get("previous_candidate_response") or "")
        )
        resume_years = float(
            (state.get("resume_context") or {}).get("experience_years") or 0
        )
        if match and resume_years > 0:
            claimed_years = float(match.group(1) or match.group(2))
            # Allow normal rounding of a fractional parsed estimate, but flag a
            # materially different whole-year claim.
            if abs(claimed_years - resume_years) > 0.5:
                violations.append(
                    _violation(
                        state,
                        "experience_years_mismatch",
                        severity="medium",
                        metadata={
                            "resume_experience_years": resume_years,
                            "claimed_experience_years": claimed_years,
                        },
                    )
                )
    return violations


async def classify_candidate_response(state: InterviewState) -> dict[str, Any]:
    """
    Classify candidate speech as an answer, silence, or clarification request.

    Bypasses the LLM for deterministic high-confidence cases. Identifies turn
    violations, decides if an answer is substantial enough to evaluate, and routes
    the graph to the appropriate next node.

    Args:
        state: The current interview state.

    Returns:
        A dictionary containing the classification results, detected violations,
        and the `next_action` routing key.
    """

    started_at = time.perf_counter()
    text = str(state.get("previous_candidate_response") or "")
    classification = _deterministic_classification(state, text)
    source = "deterministic"

    if classification is None:
        try:
            model_result = await classify_with_schema(
                _classification_messages(state),
                CandidateResponseClassification,
                key_slot=active_turn_key_slot(state),
            )
        except Exception:
            logger.exception(
                "Strict candidate-response classification failed",
                extra={
                    "candidate_assessment_id": state.get("candidate_assessment_id"),
                    "question_id": state.get("current_question_id"),
                },
            )
            model_result = _safe_fallback(text)
            source = "validated_fallback"
        else:
            source = "llm"
        classification = model_result.model_dump()

    response_type = str(classification["response_type"])
    is_substantial = classification.get("is_substantial")
    if response_type == "answer" and bool(state.get("is_self_introduction")):
        # Self-introduction elaboration is intentionally governed by one metric:
        # an answer with fewer than 15 spoken words needs more detail.
        is_substantial = (
            _spoken_word_count(text) >= SELF_INTRO_ELABORATION_WORD_THRESHOLD
        )
        classification["is_substantial"] = is_substantial
    elif (
        response_type == "answer"
        and state.get("current_section_kind") == "behavioural_cultural"
    ):
        word_count = _spoken_word_count(text)
        if word_count > BEHAVIOURAL_ANSWER_WORD_THRESHOLD:
            is_substantial = word_count > BEHAVIOURAL_SUBSTANTIAL_WORD_THRESHOLD
            classification["is_substantial"] = is_substantial

    clarification_type = classification.get("clarification_type")
    resume_skill_match = clarification_type == "skip_question" and _resume_has_skill(
        state
    )
    new_violations = _turn_violations(
        state,
        response_type=response_type,
        clarification_type=(str(clarification_type) if clarification_type else None),
        resume_skill_match=resume_skill_match,
    )
    recent_violations = [
        *list(state.get("recent_violations") or []),
        *new_violations,
    ][-20:]

    pending_candidate_turn = dict(state.get("pending_candidate_turn") or {})
    metadata = dict(pending_candidate_turn.get("metadata") or {})
    metadata.update(
        {
            "classification": classification,
            "classification_source": source,
            "clarification_type": clarification_type,
            "is_substantial": is_substantial,
        }
    )
    pending_candidate_turn.update(
        {
            "response_type": response_type,
            "metadata": metadata,
        }
    )

    silence_stage = state.get("silence_stage") or "none"
    if response_type != "silence" and classification.get("clarification_type") not in {
        "time_to_think",
        "decline_think_time",
    }:
        silence_stage = "none"

    logger.info(
        "Candidate response classified",
        extra={
            "candidate_assessment_id": state.get("candidate_assessment_id"),
            "question_id": state.get("current_question_id"),
            "response_type": response_type,
            "is_substantial": is_substantial,
            "classification_source": source,
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000, 2),
        },
    )

    return {
        "last_classification": classification,
        "classification_source": source,
        "last_response_type": response_type,
        "last_response_substantial": is_substantial,
        "last_skip_resume_skill_match": resume_skill_match,
        "pending_candidate_turn": pending_candidate_turn,
        "violations_to_persist": new_violations,
        "recent_violations": recent_violations,
        "silence_stage": silence_stage,
        "next_action": (
            "evaluate_answer"
            if (
                response_type == "answer"
                and is_substantial
                and state.get("current_section_kind") != "behavioural_cultural"
            )
            else (
                "check_time"
                if response_type == "answer" and is_substantial
                else "generate_bot_response"
            )
        ),
    }
