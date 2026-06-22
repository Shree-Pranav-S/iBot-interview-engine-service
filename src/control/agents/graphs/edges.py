"""Conditional routing for the reference-style interview graph."""

from __future__ import annotations

from src.control.agents.state import InterviewState


def route_after_session_init(state: InterviewState) -> str:
    return "await_response" if state.get("next_node") == "await_response" else "opening"


def route_after_timer_check(state: InterviewState) -> str:
    next_node = state.get("next_node")
    if next_node == "closing" or state.get("should_close"):
        return "closing"
    if next_node == "section_transition":
        return "section_transition"
    return route_response(state)


def route_response(state: InterviewState) -> str:
    return {
        "answer": "evaluate_answer",
        "clarification": "handle_clarification",
        "silence": "handle_silence",
        "irrelevant": "handle_irrelevant",
        "skip": "handle_skip",
        "think_request": "handle_silence",
        "time_up": "closing",
    }.get(
        state.get("response_class") or state.get("last_response_classification"),  # type: ignore
        "evaluate_answer",
    )


def route_after_generate(state: InterviewState) -> str:
    if state.get("next_node") == "section_transition":
        return "section_transition"
    if state.get("next_node") == "closing" or state.get("should_close"):
        return "closing"
    return "await_response"


def route_after_silence(state: InterviewState) -> str:
    if state.get("next_node") == "generate_question":
        return "generate_question"
    if state.get("next_node") == "closing" or state.get("should_close"):
        return "closing"
    return "await_response"


def route_after_irrelevant(state: InterviewState) -> str:
    if state.get("next_node") == "closing" or state.get("should_close"):
        return "closing"
    return "await_response"


def route_after_skip(state: InterviewState) -> str:
    if state.get("next_node") == "generate_question":
        return "generate_question"
    return "await_response"
