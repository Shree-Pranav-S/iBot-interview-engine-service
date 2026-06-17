"""
InterviewState — LangGraph state schema for the interview workflow.

This TypedDict carries all context across the graph lifetime. Every field
is checkpointed to Postgres via AsyncPostgresSaver after each node
execution, providing full fault tolerance and reconnection support.
"""

from __future__ import annotations

from typing import TypedDict


class SectionState(TypedDict):
    """Describes a single interview section (e.g. 'Python', 'System Design')."""

    name: str
    skill: str
    priority_score: int
    time_budget_secs: int
    time_elapsed_secs: int
    questions_asked: int
    concepts_covered: list[str]
    is_complete: bool


class InterviewState(TypedDict, total=False):
    """
    Full graph state for a single candidate interview session.

    Fields marked total=False so LangGraph can do partial updates
    (each node returns only the delta it wants to merge).
    """

    # ── Static context (loaded once at init) ──────────────────────────────────
    candidate_assessment_id: str
    interview_plan: dict  # sections, time allocations, order
    jd_analysis: dict  # skill priorities, behavioural signals
    resume_context: dict  # parsed resume from LlamaParse

    # ── Section tracking ──────────────────────────────────────────────────────
    sections: list[SectionState]  # ordered list of section objects
    current_section_index: int
    current_section_time_remaining_secs: int

    # ── Turn tracking ─────────────────────────────────────────────────────────
    turn_number: int
    current_question_text: str
    current_question_difficulty: str  # easy|medium|hard
    consecutive_strong: int  # for escalation logic
    consecutive_weak: int  # for recalibration logic
    used_concepts: list[str]  # anti-repetition injection
    irrelevant_strike_count: int  # 0,1,2 → 3 = terminate

    # ── Silence sub-state ─────────────────────────────────────────────────────
    silence_attempt: int  # 0 = first nudge, 1 = think offer, 2 = zero
    think_timer_active: bool

    # ── Per-turn answer data ──────────────────────────────────────────────────
    last_transcript: str
    last_response_classification: str  # answer|clarification|silence|irrelevant
    last_evaluation: dict | None

    # ── Accumulated turn records ──────────────────────────────────────────────
    transcript_turns: list[dict]
    answer_evaluations: list[dict]

    # ── Bot output (read by WebSocket layer) ──────────────────────────────────
    bot_reply_text: str  # latest bot utterance for TTS delivery
    bot_reply_type: str  # opening|question|clarification|nudge|transition|closing

    # ── Session control ───────────────────────────────────────────────────────
    session_status: str  # in_progress|paused|completed|deactivated|terminated
    timer_started_at: str  # ISO timestamp
    total_elapsed_secs: int
    total_pause_secs: int
    paused_at: str | None
    grace_period_expires_at: str | None
    auto_submit_triggered: bool
