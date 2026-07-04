"""State contract for the phase-three live interview workflow."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

ResponseType = Literal["answer", "clarification", "irrelevant", "silence"]
Difficulty = Literal["easy", "medium", "hard"]
SilenceStage = Literal[
    "none",
    "awaiting_think_confirmation",
    "thinking",
    "nudged",
]


class InterviewState(TypedDict, total=False):
    """Checkpointed data shared by the phase-three LangGraph nodes."""

    # Identity and prerequisite context.
    candidate_assessment_id: str
    interview_session_id: str
    candidate_name: str
    company_name: str
    resume_context: dict[str, Any]
    inferred_difficulty: str
    runtime_sections: list[dict[str, Any]]

    # Active section and question.
    current_section_index: int
    current_section_kind: Literal[
        "self_intro",
        "technical",
        "behavioural_cultural",
    ]
    current_technical_skill: str | None
    current_question_id: str
    current_question_text: str
    last_rephrased_question: str | None
    current_question_difficulty: Difficulty | None

    # LiveKit input for the current turn.
    candidate_event: dict[str, Any] | str | None
    previous_candidate_response: str
    speculative_interviewer_result: dict[str, Any] | None
    # Stable key-pool affinity captured before this candidate turn is processed.
    # It survives candidate persistence so regeneration/rephrasing stays on the
    # same organization even after ``turn_number`` advances.
    llm_key_slot: int | None

    # Strict model outputs, stored as JSON-compatible dictionaries.
    last_classification: dict[str, Any] | None
    classification_source: str | None
    last_response_type: ResponseType | None
    last_response_substantial: bool | None
    latest_evaluation: dict[str, Any] | None

    # Question history and deterministic adaptation.
    asked_questions: list[dict[str, Any]]
    used_topics_by_skill: dict[str, list[str]]
    skill_evaluation_streaks: dict[str, dict[str, int]]
    question_variation_seed: str
    probe_deeper: bool
    thread_follow_up_used: bool
    last_skip_resume_skill_match: bool

    # Static/dynamic response control.
    bot_reply_text: str
    bot_reply_type: str
    response_preface_text: str | None
    # Merged interviewer-turn clarification and closing handoff.
    pregenerated_closing_lead: str | None
    pending_clarification_text: str | None
    silence_stage: SilenceStage
    skip_attempts_for_current_question: int
    self_intro_elaboration_requested: bool
    self_intro_accumulated_response: str
    should_advance_question: bool
    phase_complete: bool
    next_action: str | None

    # One bot/candidate interaction shares a turn number.
    turn_number: int
    pending_bot_turn: dict[str, Any] | None
    pending_candidate_turn: dict[str, Any] | None
    violations_to_persist: list[dict[str, Any]]
    recent_violations: list[dict[str, Any]]

    # Total and section timing.
    total_duration_secs: int
    elapsed_secs: int
    section_budgets_secs: dict[str, int]
    current_section_budget_secs: int
    current_section_started_elapsed_secs: int
    current_section_elapsed_secs: int
    current_section_remaining_secs: int
    pending_section_index: int | None
    suppress_previous_context_for_next_question: bool
    transition_reason: str | None
    barge_in_triggered: bool
    self_intro_leftover_redistributed: bool

    # Timer/lifecycle metadata. Initialization never starts the timer.
    timer_started: bool
    timer_started_at: str | None
    session_status: str
    should_close: bool
    closing_done: bool
    closing_reason: str | None
    holistic_evaluation_status: str | None
    holistic_evaluation_task_id: str | None
