"""LangGraph state for the voice-led interview workflow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

SessionStatus = Literal[
    "INITIALIZING",
    "IN_PROGRESS",
    "PAUSED",
    "COMPLETED",
    "EVALUATED",
    "DEACTIVATED",
    "TERMINATED",
]
Difficulty = Literal["easy", "medium", "hard"]
ResponseType = Literal[
    "answer",
    "clarification",
    "silence",
    "irrelevant",
    "non_answer",
    "technical_issue",
    "interruption",
    "disconnect",
    "skip",
    "think_request",
    "no_think",
    "timer_expired",
]


class InterviewState(TypedDict, total=False):
    # Identity
    candidate_assessment_id: str
    interview_session_id: str | None
    thread_id: str

    # Context
    candidate_name: str
    candidate_email: str | None
    role_name: str
    company_name: str
    assessment_title: str
    resume_parsed: dict[str, Any]
    focus_areas: dict[str, Any] | list[dict[str, Any]] | None
    interview_plan: dict[str, Any]
    runtime_sections: list[dict[str, Any]]
    interview_duration_mins: int

    # Time
    total_duration_secs: int
    started_at: str | None
    elapsed_secs: int
    total_pause_secs: int
    section_started_at: str | None
    current_section_elapsed_secs: int
    current_section_remaining_secs: int
    remaining_secs: int
    soft_total_overrun_secs: int
    soft_section_overrun_secs: int
    force_close_due_to_overrun: bool
    force_transition_due_to_overrun: bool
    force_behavioural_cultural_due_to_time: bool
    close_after_behavioural_cultural: bool
    section_budgets: dict[str, int]
    section_order: list[str]

    # Current position
    current_section: str
    current_section_index: int
    current_section_name: str
    current_skill: str | None
    current_question_id: str | None
    current_question_text: str | None
    current_difficulty: Difficulty
    turn_number: int

    # Short-term memory
    recent_turns: list[dict[str, Any]]
    last_candidate_event: dict[str, Any] | None
    last_response_type: ResponseType | None
    last_response_substantial: bool | None
    last_response_reason: str | None
    last_classification: dict[str, Any] | None
    local_route: dict[str, Any] | None
    live_decision: dict[str, Any] | None
    last_answer_strength: str | None
    last_bot_text: str | None
    bot_reply_text: str
    bot_reply_type: str

    # Adaptive control
    asked_questions: list[dict[str, Any]]
    skill_progress: dict[str, Any]
    live_evaluations: list[dict[str, Any]]
    latest_evaluation: dict[str, Any] | None
    expected_signals: list[str]
    clarification_count_for_current_question: int
    silence_count_for_current_question: int
    non_answer_count_for_current_question: int
    skip_count_for_current_question: int
    think_silence_count: int
    awaiting_think_confirmation: bool
    think_extension_active: bool
    irrelevant_count_total: int

    # Session reliability
    session_status: SessionStatus
    disconnected_at: str | None
    grace_period_expires_at: str | None
    reconnect_count: int

    # Routing and node-local durable writes
    next_action: str | None
    next_node: str | None
    should_close: bool
    closing_done: bool
    final_evaluation_status: str | None
    holistic_evaluation_task_id: str | None
    resumed: bool
    candidate_event: dict[str, Any] | str | None
    normalized_candidate_event: dict[str, Any] | None
    pending_bot_turn: dict[str, Any] | None
    pending_candidate_turn: dict[str, Any] | None
    violation_to_persist: dict[str, Any] | None
    next_section_index: int | None
    previous_section: str | None
    closing_reason: str | None
