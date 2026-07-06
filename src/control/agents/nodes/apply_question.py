"""Apply a staged or generated question to interview state without an LLM call."""

from __future__ import annotations

from typing import Any, cast

from src.control.agents.nodes.question_strategy import target_section
from src.control.agents.nodes.turn_utils import build_bot_turn
from src.control.agents.state import Difficulty, InterviewState


def apply_resolved_question(
    state: InterviewState,
    *,
    acknowledgement: str,
    question_text: str,
    topic: str,
    current_skill: str | None,
    current_difficulty: Difficulty | None,
    probe_deeper: bool,
    follow_interesting_thread: bool = False,
) -> dict[str, Any]:
    """
    Commit a validated question to state and produce bot speech fields.

    Shared by ``interviewer_turn`` (hot path, skips ``generate_next_question``)
    and ``generate_next_question`` (fallback / barge-in preface paths).
    Core API transcript persistence remains non-blocking in ``persist_turn``;
    skipping the extra graph node here reduces LangGraph checkpoint latency.

    Args:
        state: Current interview state (timing updates should already be merged).
        acknowledgement: Spoken bridge before the question.
        question_text: The next interview question.
        topic: Short label for deduplication tracking.
        current_skill: Active technical skill, or None for behavioural.
        current_difficulty: Enforced difficulty for technical questions.
        probe_deeper: Whether this question probes the prior weak answer.
        follow_interesting_thread: True when holding difficulty for a curiosity follow-up.

    Returns:
        State updates including ``bot_reply_text`` and ``pending_bot_turn``.
    """

    section_index, section = target_section(state)
    section_kind = str(section.get("section_kind") or "technical")
    section_name = str(section.get("section_name") or "technical")
    entering_new_section = (
        section_index != int(state.get("current_section_index") or 0)
        or state.get("pending_section_index") is not None
    )

    if follow_interesting_thread and not entering_new_section:
        held = state.get("current_question_difficulty")
        if held is not None:
            current_difficulty = held

    spoken_text = f"{acknowledgement} {question_text}".strip()
    bot_reply_type = (
        "self_intro_transition_question"
        if state.get("transition_reason") == "self_introduction_complete"
        else "new_question"
    )
    runtime_sections = list(state.get("runtime_sections") or [])
    section_budgets = {
        str(key): int(value)
        for key, value in (state.get("section_budgets_secs") or {}).items()
    }
    redistributed_self_intro_leftover = bool(
        state.get("self_intro_leftover_redistributed")
    )
    previous_section_index = int(state.get("current_section_index") or 0)
    if (
        entering_new_section
        and state.get("current_section_kind") == "self_intro"
        and not redistributed_self_intro_leftover
    ):
        remaining_intro_secs = max(
            0, int(state.get("current_section_remaining_secs") or 0)
        )
        recipients = [
            index
            for index, future_section in enumerate(runtime_sections)
            if index > previous_section_index
            and future_section.get("section_kind")
            in {"technical", "behavioural_cultural"}
        ]
        if remaining_intro_secs > 0 and recipients:
            share, remainder = divmod(remaining_intro_secs, len(recipients))
            for pos, idx in enumerate(recipients):
                bonus = share + (1 if pos < remainder else 0)
                section_budgets[str(idx)] = max(
                    1,
                    int(section_budgets.get(str(idx), 60)) + bonus,
                )
        redistributed_self_intro_leftover = True

    section_budget = int(section_budgets.get(str(section_index), 60))
    section_started_elapsed = (
        int(state.get("elapsed_secs") or 0)
        if entering_new_section
        else int(state.get("current_section_started_elapsed_secs") or 0)
    )
    section_elapsed = max(
        0,
        int(state.get("elapsed_secs") or 0) - section_started_elapsed,
    )
    future_reserved = sum(
        max(0, int(section_budgets.get(str(index)) or 0))
        for index, future_section in enumerate(runtime_sections)
        if index > section_index
        and future_section.get("section_kind")
        in {
            "technical",
            "behavioural_cultural",
        }
    )
    total_duration = max(1, int(state.get("total_duration_secs") or 1))
    protected_deadline = (
        total_duration - future_reserved if future_reserved > 0 else total_duration
    )
    section_deadline = max(
        0,
        min(section_started_elapsed + section_budget, protected_deadline),
    )
    section_remaining = max(
        0,
        min(
            section_budget - section_elapsed,
            section_deadline - int(state.get("elapsed_secs") or 0),
        ),
    )

    if entering_new_section:
        thread_follow_up_used = False
    elif follow_interesting_thread:
        thread_follow_up_used = True
    else:
        thread_follow_up_used = False

    question_id = (
        f"{state['interview_session_id']}:question:"
        f"{len(state.get('asked_questions') or []) + 1}"
    )
    section_updates: dict[str, Any] = {
        "current_section_index": section_index,
        "current_section_kind": section_kind,
        "current_technical_skill": current_skill,
        "current_question_id": question_id,
        "current_question_text": question_text,
        "last_rephrased_question": None,
        "current_question_difficulty": current_difficulty,
        "current_section_budget_secs": section_budget,
        "current_section_started_elapsed_secs": section_started_elapsed,
        "current_section_elapsed_secs": section_elapsed,
        "current_section_remaining_secs": section_remaining,
        "section_budgets_secs": section_budgets,
        "self_intro_leftover_redistributed": redistributed_self_intro_leftover,
    }
    next_state = cast(InterviewState, {**state, **section_updates})
    pending_bot_turn = build_bot_turn(
        next_state,
        text=spoken_text,
        question_type="new_question",
        question_text=question_text,
        acknowledgement=acknowledgement,
        topic=topic,
    )
    question_record = {
        "question_id": question_id,
        "question_text": question_text,
        "difficulty": current_difficulty,
        "skill": current_skill,
        "section": section_name,
        "topic": topic,
        "acknowledgement": acknowledgement,
    }
    asked_questions = [
        *list(state.get("asked_questions") or []),
        question_record,
    ]
    used_topics = {
        key: list(value)
        for key, value in (state.get("used_topics_by_skill") or {}).items()
    }
    if current_skill:
        used_topics.setdefault(current_skill.casefold(), []).append(topic)

    return {
        **section_updates,
        "probe_deeper": probe_deeper,
        "thread_follow_up_used": thread_follow_up_used,
        "asked_questions": asked_questions,
        "used_topics_by_skill": used_topics,
        "pending_section_index": None,
        "suppress_previous_context_for_next_question": False,
        "transition_reason": None,
        "barge_in_triggered": False,
        "pending_clarification_text": None,
        "bot_reply_text": spoken_text,
        "bot_reply_type": bot_reply_type,
        "pending_bot_turn": pending_bot_turn,
        "response_preface_text": None,
        "skip_attempts_for_current_question": 0,
        "self_intro_elaboration_requested": False,
        "self_intro_accumulated_response": "",
        "should_advance_question": False,
        "phase_complete": False,
        "latest_evaluation": None,
        "next_action": "await_candidate_response",
        "silence_stage": "none",
    }
