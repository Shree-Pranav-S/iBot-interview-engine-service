"""LangGraph assembly for the LiveKit-owned interview workflow."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from src.config.settings import settings
from src.control.agents.nodes.await_response import (
    await_candidate_response_interrupt,
)
from src.control.agents.nodes.classify_response import classify_response_node
from src.control.agents.nodes.closing import generate_closing_message
from src.control.agents.nodes.context_utils import is_technical_section
from src.control.agents.nodes.decide_next_action import decide_next_action
from src.control.agents.nodes.edge_handlers import (
    handle_clarification,
    handle_disconnect,
    handle_irrelevant,
    handle_non_answer,
    handle_silence,
    handle_think_request,
)
from src.control.agents.nodes.evaluation import live_evaluate_answer_node
from src.control.agents.nodes.final_evaluation import final_evaluation
from src.control.agents.nodes.generate_question import generate_question
from src.control.agents.nodes.init_session import init_or_resume_session
from src.control.agents.nodes.opening import generate_opening_message
from src.control.agents.nodes.persist_turn import persist_interview_turn
from src.control.agents.nodes.section_transition import generate_section_transition
from src.control.agents.nodes.time_manager import check_time_budget
from src.control.agents.nodes.update_skill_state import update_skill_state
from src.control.agents.state import InterviewState

_graph: Any | None = None
_checkpointer_cm: Any | None = None
_checkpointer: Any | None = None


def _checkpoint_uri(db_uri: str) -> str:
    return db_uri.replace("postgresql+asyncpg://", "postgresql://", 1)


def _route_after_persist_interview_turn(state: InterviewState) -> str:
    next_node = state.get("next_node")
    if next_node == "end":
        return "end"
    if next_node in {
        "check_time_budget",
        "await_candidate_response_interrupt",
        "final_evaluation",
        "generate_closing_message",
    }:
        return str(next_node)
    if state.get("should_close") and state.get("bot_reply_type") == "closing":
        return "final_evaluation"
    if state.get("should_close") or state.get("next_action") == "complete":
        return "generate_closing_message"
    if state.get("bot_reply_type") == "question":
        return "await_candidate_response_interrupt"
    return "check_time_budget"


def _route_after_time_budget(state: InterviewState) -> str:
    next_action = state.get("next_action")
    if state.get("should_close") or next_action == "complete":
        return "generate_closing_message"
    if next_action == "section_transition":
        return "generate_section_transition"
    return "generate_question"


def _route_after_classification(state: InterviewState) -> str:
    response_type = state.get("last_response_type") or "answer"
    if response_type == "answer":
        if state.get("last_response_substantial") is False:
            return "handle_non_answer"
        if is_technical_section(state):
            return "live_evaluate_answer"
        return "decide_next_action"
    if response_type == "clarification_question":
        return "handle_clarification"
    if response_type == "silence":
        return "handle_silence"
    if response_type == "irrelevant_answer":
        return "handle_irrelevant"
    if response_type == "think_request":
        return "handle_think_request"
    if response_type == "disconnect":
        return "handle_disconnect"
    if response_type == "timer_expired":
        return "check_time_budget"
    return "check_time_budget"


def _route_after_edge_handler(state: InterviewState) -> str:
    if (
        state.get("pending_candidate_turn")
        or state.get("pending_bot_turn")
        or state.get("violation_to_persist")
    ):
        return "persist_interview_turn"
    if state.get("next_node") == "end":
        return "end"
    if state.get("next_node") == "await_candidate_response_interrupt":
        return "await_candidate_response_interrupt"
    if state.get("should_close") or state.get("next_action") == "complete":
        return "generate_closing_message"
    return "check_time_budget"


def build_interview_graph(checkpointer: Any | None = None) -> Any:
    builder = StateGraph(InterviewState)

    builder.add_node("init_or_resume_session", init_or_resume_session)
    builder.add_node("generate_opening_message", generate_opening_message)
    builder.add_node("persist_interview_turn", persist_interview_turn)
    builder.add_node("check_time_budget", check_time_budget)
    builder.add_node("generate_question", generate_question)
    builder.add_node("generate_section_transition", generate_section_transition)
    builder.add_node(
        "await_candidate_response_interrupt",
        await_candidate_response_interrupt,
    )
    builder.add_node("classify_response", classify_response_node)
    builder.add_node("handle_clarification", handle_clarification)
    builder.add_node("handle_silence", handle_silence)
    builder.add_node("handle_non_answer", handle_non_answer)
    builder.add_node("handle_irrelevant", handle_irrelevant)
    builder.add_node("handle_think_request", handle_think_request)
    builder.add_node("handle_disconnect", handle_disconnect)
    builder.add_node("live_evaluate_answer", live_evaluate_answer_node)
    builder.add_node("update_skill_state", update_skill_state)
    builder.add_node("decide_next_action", decide_next_action)
    builder.add_node("generate_closing_message", generate_closing_message)
    builder.add_node("final_evaluation", final_evaluation)

    builder.add_edge(START, "init_or_resume_session")
    builder.add_edge("init_or_resume_session", "generate_opening_message")
    builder.add_edge("generate_opening_message", "persist_interview_turn")
    builder.add_conditional_edges(
        "persist_interview_turn",
        _route_after_persist_interview_turn,
        {
            "check_time_budget": "check_time_budget",
            "await_candidate_response_interrupt": "await_candidate_response_interrupt",
            "final_evaluation": "final_evaluation",
            "generate_closing_message": "generate_closing_message",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "check_time_budget",
        _route_after_time_budget,
        {
            "generate_question": "generate_question",
            "generate_section_transition": "generate_section_transition",
            "generate_closing_message": "generate_closing_message",
        },
    )
    builder.add_edge("generate_question", "persist_interview_turn")
    builder.add_edge("generate_section_transition", "persist_interview_turn")
    builder.add_edge("await_candidate_response_interrupt", "classify_response")
    builder.add_conditional_edges(
        "classify_response",
        _route_after_classification,
        {
            "live_evaluate_answer": "live_evaluate_answer",
            "handle_clarification": "handle_clarification",
            "handle_silence": "handle_silence",
            "handle_non_answer": "handle_non_answer",
            "handle_irrelevant": "handle_irrelevant",
            "handle_think_request": "handle_think_request",
            "handle_disconnect": "handle_disconnect",
            "decide_next_action": "decide_next_action",
            "check_time_budget": "check_time_budget",
        },
    )
    for node_name in (
        "handle_clarification",
        "handle_silence",
        "handle_non_answer",
        "handle_irrelevant",
        "handle_think_request",
        "handle_disconnect",
    ):
        builder.add_conditional_edges(
            node_name,
            _route_after_edge_handler,
            {
                "persist_interview_turn": "persist_interview_turn",
                "check_time_budget": "check_time_budget",
                "await_candidate_response_interrupt": "await_candidate_response_interrupt",
                "generate_closing_message": "generate_closing_message",
                "end": END,
            },
        )
    builder.add_edge("live_evaluate_answer", "update_skill_state")
    builder.add_edge("update_skill_state", "decide_next_action")
    builder.add_edge("decide_next_action", "check_time_budget")
    builder.add_edge("generate_closing_message", "persist_interview_turn")
    builder.add_edge("final_evaluation", END)

    return builder.compile(checkpointer=checkpointer)


async def init_graph(db_uri: str | None = None) -> Any:
    global _checkpointer, _checkpointer_cm, _graph

    if _graph is not None:
        return _graph

    _checkpointer_cm = AsyncPostgresSaver.from_conn_string(
        _checkpoint_uri(db_uri or settings.DATABASE_URL)
    )
    _checkpointer = await _checkpointer_cm.__aenter__()
    await _checkpointer.setup()
    _graph = build_interview_graph(checkpointer=_checkpointer)
    return _graph


async def get_graph() -> Any:
    if _graph is None:
        return await init_graph()
    return _graph


async def close_graph() -> None:
    global _checkpointer, _checkpointer_cm, _graph

    if _checkpointer_cm is not None:
        await _checkpointer_cm.__aexit__(None, None, None)
    _checkpointer = None
    _checkpointer_cm = None
    _graph = None
