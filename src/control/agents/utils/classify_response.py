"""Deterministic fast paths for candidate response routing."""

from __future__ import annotations

import re
from typing import Any

from src.control.agents.state import InterviewState
from src.control.agents.utils.question_strategy import is_self_intro_phase
from src.utils.interview_graph import deterministic_violation_id, utc_now_iso

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
TECHNICAL_ANSWER_SIGNAL_PATTERN = re.compile(
    r"\b(?:"
    r"api|algorithm|async|cache|class|code|connection|cpu|database|db|"
    r"deadlock|debug|diagnos(?:e|ed|ing|is|tic)|docker|endpoint|exception|"
    r"framework|function|gil|hash|index|latency|lock(?:ed|ing|s)?|"
    r"memory|mutex(?:es)?|object|postgres(?:ql)?|process|production|"
    r"python|quer(?:y|ies)|queue|redis|request|response|schema|server|"
    r"sql|thread(?:ed|ing|s)?|transaction(?:al|s)?|worker"
    r")\b",
    re.IGNORECASE,
)
QUESTION_KEYWORD_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "answer",
        "asked",
        "causing",
        "could",
        "describe",
        "does",
        "from",
        "have",
        "into",
        "interview",
        "question",
        "should",
        "take",
        "tell",
        "that",
        "their",
        "there",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
        "your",
    }
)


def _spoken_word_count(text: str) -> int:
    """Count spoken words using the same technical-token rules as classification."""

    return len(re.findall(r"\b[\w+#.-]+\b", text))


def _keyword_tokens(text: str) -> set[str]:
    """Return content-bearing tokens for loose question/answer overlap checks."""

    return {
        token.casefold()
        for token in re.findall(r"\b[\w+#.-]+\b", text)
        if len(token) >= 4 and token.casefold() not in QUESTION_KEYWORD_STOPWORDS
    }


def _has_loose_token_overlap(question: str, response: str) -> bool:
    """Allow simple morphology such as deadlock/lock when judging topicality."""

    question_tokens = _keyword_tokens(question)
    response_tokens = _keyword_tokens(response)
    for question_token in question_tokens:
        for response_token in response_tokens:
            if question_token == response_token:
                return True
            if min(len(question_token), len(response_token)) < 4:
                continue
            if question_token in response_token or response_token in question_token:
                return True
    return False


def technical_answer_signal_is_present(state: InterviewState, text: str) -> bool:
    """
    Detect weak but topical technical answer attempts that must not be violations.

    This intentionally does not grade correctness. It only protects routing when a
    candidate uses technical vocabulary or overlaps with the active question.
    """

    normalized = " ".join(str(text or "").split())
    if state.get("current_section_kind") != "technical":
        return False
    if _spoken_word_count(normalized) < 6:
        return False
    if IRRELEVANT_TOPIC_PATTERN.search(normalized):
        return False
    if normalized.rstrip().endswith("?") or QUESTION_LIKE_ANSWER_PATTERN.search(
        normalized
    ):
        return False

    question = str(state.get("current_question_text") or "")
    current_skill = str(state.get("current_technical_skill") or "").strip()
    if current_skill and re.search(
        rf"\b{re.escape(current_skill)}\b",
        normalized,
        re.IGNORECASE,
    ):
        return True
    return bool(TECHNICAL_ANSWER_SIGNAL_PATTERN.search(normalized)) or (
        bool(question) and _has_loose_token_overlap(question, normalized)
    )


def _self_intro_combined_text(state: InterviewState, text: str) -> str:
    """Merge prior self-intro speech with the current turn for substantiality checks."""

    prior = " ".join(str(state.get("self_intro_accumulated_response") or "").split())
    current = " ".join(str(text or "").split())
    if not prior:
        return current
    if not current:
        return prior
    if current.casefold().startswith(prior.casefold()):
        return current
    return f"{prior} {current}".strip()


def self_intro_is_substantial(state: InterviewState, text: str) -> bool:
    """Return True when cumulative self-intro speech meets the elaboration threshold."""

    return (
        _spoken_word_count(_self_intro_combined_text(state, text))
        >= SELF_INTRO_ELABORATION_WORD_THRESHOLD
    )


def is_deterministic_classification_source(source: str | None) -> bool:
    """Return True when a turn was classified without the classifier LLM."""

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
    if det is not None and det.get("response_type") in {
        "silence",
        "clarification",
        "irrelevant",
    }:
        return True
    if is_self_intro_phase(state):
        return True
    word_count = _spoken_word_count(text)
    return (
        state.get("current_section_kind") == "behavioural_cultural"
        and word_count > BEHAVIOURAL_ANSWER_WORD_THRESHOLD
    )


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
        The dictionary matching the interviewer-turn classification structure.
    """
    return {
        "response_type": response_type,
        "clarification_type": clarification_type,
        "is_substantial": is_substantial,
        "question_doubt_response": question_doubt_response,
    }


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
        if is_self_intro_phase(state) and self_intro_is_substantial(
            state,
            "",
        ):
            return _result(
                response_type="answer",
                clarification_type=None,
                is_substantial=True,
            )
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

    if is_self_intro_phase(state):
        return _result(
            response_type="answer",
            clarification_type=None,
            is_substantial=self_intro_is_substantial(state, text),
        )
    if state.get("current_section_kind") == "behavioural_cultural":
        word_count = _spoken_word_count(text)
        if word_count > BEHAVIOURAL_ANSWER_WORD_THRESHOLD:
            return _result(
                response_type="answer",
                clarification_type=None,
                is_substantial=(word_count > BEHAVIOURAL_SUBSTANTIAL_WORD_THRESHOLD),
            )
    if IRRELEVANT_TOPIC_PATTERN.search(text):
        return _result(
            response_type="irrelevant",
            clarification_type=None,
            is_substantial=None,
        )
    return None


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
