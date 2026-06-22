"""LangGraph state for the live interview workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

SectionName = str
ResponseClass = Literal[
    "answer",
    "clarification",
    "silence",
    "irrelevant",
    "skip",
    "think_request",
    "time_up",
]


class SectionState(TypedDict, total=False):
    name: str
    section_name: str
    skill: str | None
    priority_score: float | None
    allocated_mins: float
    time_budget_secs: int
    time_elapsed_secs: int
    questions_asked: int
    concepts_covered: list[str]
    is_complete: bool


class QuestionScore(TypedDict, total=False):
    question: str
    section: SectionName
    skill: str | None
    concept: str | None
    difficulty: str
    quality: str
    raw_score: float
    reasoning: str
    signals_demonstrated: list[str]
    signals_missing: list[str]
    turn_number: int
    nudge_given: bool


class Violation(TypedDict, total=False):
    turn_number: int
    violation_type: Literal[
        "irrelevant",
        "resume_mismatch",
        "yoe_mismatch",
        "terminated",
        "silence",
    ]
    candidate_transcript: str
    timestamp: float


class InterviewState(TypedDict, total=False):
    # Static interview context
    candidate_assessment_id: str
    role_name: str
    company_name: str
    interview_plan: dict
    resume_parsed: dict
    resume_context: dict

    # Section and timer tracking
    sections: list[SectionState]
    current_section_index: int
    current_section_name: SectionName
    current_section_time_remaining_secs: int
    section_started_at: float
    section_allocated_secs: float
    total_interview_allocated_secs: int
    timer_started_at: str
    interview_started_at: float
    total_elapsed_secs: int
    total_pause_secs: int
    paused_at: str | None
    grace_period_expires_at: str | None
    auto_submit_triggered: bool

    # Question and turn tracking
    turn_number: int
    current_question: str
    current_question_text: str
    current_question_difficulty: str
    current_question_concept: str
    last_bot_text: str
    last_transcript: str
    last_response_classification: ResponseClass | None
    used_concepts: list[str]
    concepts_covered_in_section: list[str]
    questions_asked_in_section: int
    irrelevant_count: int
    irrelevant_strike_count: int
    current_difficulty_level: int
    next_question_mode: str
    last_question_was_weak_retry: bool

    # Per-turn candidate data
    candidate_raw_text: str
    candidate_stt_confidence: float | None
    response_class: ResponseClass | None

    # Silence and skip handling
    awaiting_think_decision: bool
    think_timer_active: bool
    skip_requested: bool

    # Accumulated records
    transcript: Annotated[list[dict], operator.add]
    question_scores: Annotated[list[QuestionScore], operator.add]
    violations: Annotated[list[Violation], operator.add]
    last_evaluation: dict | None

    # Bot output consumed by the WebSocket layer
    bot_reply_text: str
    bot_reply_type: str

    # Session control
    session_id: str
    session_status: str
    should_close: bool
    closing_done: bool
    holistic_evaluation_done: bool
    next_node: str | None
    resumed: bool
