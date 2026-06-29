"""Phase-three LangGraph assembly and PostgreSQL checkpoint lifecycle."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from src.config.settings import settings
from src.control.agents.nodes.await_response import await_candidate_response
from src.control.agents.nodes.classify_response import (
    classify_candidate_response,
)
from src.control.agents.nodes.closing import generate_closing_message
from src.control.agents.nodes.evaluation import evaluate_substantial_answer
from src.control.agents.nodes.final_evaluation import (
    trigger_final_evaluation,
)
from src.control.agents.nodes.generate_bot_response import generate_bot_response
from src.control.agents.nodes.generate_question import generate_next_question
from src.control.agents.nodes.initialize_context import (
    initialize_interview_context,
)
from src.control.agents.nodes.opening import deliver_opening
from src.control.agents.nodes.persist_turn import (
    drain_background_persistence,
    persist_bot_output,
    persist_candidate_output,
)
from src.control.agents.nodes.time_manager import (
    check_time_after_substantial_answer,
    force_section_time_barge_in,
)
from src.control.agents.state import InterviewState

_graph: Any | None = None
_checkpointer_context: Any | None = None
_checkpointer: Any | None = None


def _checkpoint_uri(database_url: str) -> str:
    """
    Format the SQLAlchemy connection string for the AsyncPostgresSaver.

    Args:
        database_url: The raw database connection string.

    Returns:
        The formatted URI.
    """
    return database_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _route_after_classification(state: InterviewState) -> str:
    """Route to evaluation if answer is substantial, else directly to persistence."""
    if state.get("next_action") == "evaluate_answer":
        return "evaluate_substantial_answer"
    return "persist_candidate_output"


def _route_after_await(state: InterviewState) -> str:
    """Route to barge-in handler if timed out, otherwise proceed to classification."""
    if state.get("next_action") == "force_section_time_barge_in":
        return "force_section_time_barge_in"
    return "classify_candidate_response"


def _route_after_candidate_persistence(state: InterviewState) -> str:
    """Route to time check after an answer, else directly generate bot response."""
    if state.get("next_action") == "check_time":
        return "check_time_after_substantial_answer"
    return "generate_bot_response"


def _route_after_static_response(state: InterviewState) -> str:
    """Route to question generation if requested, else persist the static response."""
    if state.get("next_action") == "generate_question":
        return "generate_next_question"
    return "persist_bot_output"


def _route_after_time_check(state: InterviewState) -> str:
    """Route to closing if interview is over, otherwise generate next question."""
    if state.get("next_action") == "generate_closing":
        return "generate_closing_message"
    return "generate_next_question"


def _route_after_time_barge_in(state: InterviewState) -> str:
    """Route based on what the barge-in handler decided to force."""
    if state.get("next_action") == "generate_closing":
        return "generate_closing_message"
    if state.get("next_action") == "await_candidate_response":
        return "await_candidate_response"
    return "generate_next_question"


def _route_after_bot_persistence(state: InterviewState) -> str:
    """Route to wait for the user, unless the interview is scheduled to close."""
    if state.get("should_close") or state.get("next_action") == "end":
        return "final_evaluation"
    return "await_candidate_response"


def build_interview_graph(checkpointer: Any | None = None) -> Any:
    """
    Build the timed, section-aware interview workflow graph.

    Constructs the directed state graph defining the cyclical process of
    speaking, waiting, classifying, evaluating, timing, and generating logic.

    Args:
        checkpointer: The LangGraph checkpointer instance for state persistence.

    Returns:
        The compiled StateGraph ready for execution.
    """

    builder = StateGraph(InterviewState)
    builder.add_node("initialize_interview_context", initialize_interview_context)
    builder.add_node("deliver_opening", deliver_opening)
    builder.add_node("await_candidate_response", await_candidate_response)
    builder.add_node("classify_candidate_response", classify_candidate_response)
    builder.add_node("evaluate_substantial_answer", evaluate_substantial_answer)
    builder.add_node("persist_candidate_output", persist_candidate_output)
    builder.add_node(
        "check_time_after_substantial_answer",
        check_time_after_substantial_answer,
    )
    builder.add_node(
        "force_section_time_barge_in",
        force_section_time_barge_in,
    )
    builder.add_node("generate_bot_response", generate_bot_response)
    builder.add_node("generate_next_question", generate_next_question)
    builder.add_node("generate_closing_message", generate_closing_message)
    builder.add_node("persist_bot_output", persist_bot_output)
    builder.add_node("final_evaluation", trigger_final_evaluation)

    builder.add_edge(START, "initialize_interview_context")
    builder.add_edge("initialize_interview_context", "deliver_opening")
    builder.add_edge("deliver_opening", "persist_bot_output")
    builder.add_conditional_edges(
        "persist_bot_output",
        _route_after_bot_persistence,
        {
            "await_candidate_response": "await_candidate_response",
            "final_evaluation": "final_evaluation",
        },
    )
    builder.add_conditional_edges(
        "await_candidate_response",
        _route_after_await,
        {
            "classify_candidate_response": "classify_candidate_response",
            "force_section_time_barge_in": "force_section_time_barge_in",
        },
    )
    builder.add_conditional_edges(
        "classify_candidate_response",
        _route_after_classification,
        {
            "evaluate_substantial_answer": "evaluate_substantial_answer",
            "persist_candidate_output": "persist_candidate_output",
        },
    )
    builder.add_edge("evaluate_substantial_answer", "persist_candidate_output")
    builder.add_conditional_edges(
        "persist_candidate_output",
        _route_after_candidate_persistence,
        {
            "check_time_after_substantial_answer": (
                "check_time_after_substantial_answer"
            ),
            "generate_bot_response": "generate_bot_response",
        },
    )
    builder.add_conditional_edges(
        "check_time_after_substantial_answer",
        _route_after_time_check,
        {
            "generate_next_question": "generate_next_question",
            "generate_closing_message": "generate_closing_message",
        },
    )
    builder.add_conditional_edges(
        "force_section_time_barge_in",
        _route_after_time_barge_in,
        {
            "generate_next_question": "generate_next_question",
            "generate_closing_message": "generate_closing_message",
            "await_candidate_response": "await_candidate_response",
        },
    )
    builder.add_conditional_edges(
        "generate_bot_response",
        _route_after_static_response,
        {
            "generate_next_question": "generate_next_question",
            "persist_bot_output": "persist_bot_output",
        },
    )
    builder.add_edge("generate_next_question", "persist_bot_output")
    builder.add_edge("generate_closing_message", "persist_bot_output")
    builder.add_edge("final_evaluation", END)
    return builder.compile(checkpointer=checkpointer)


async def init_graph(database_url: str | None = None) -> Any:
    """
    Initialize the singleton graph and durable PostgreSQL checkpointer.

    Args:
        database_url: Optional connection string override.

    Returns:
        The initialized graph instance.
    """

    global _checkpointer, _checkpointer_context, _graph
    if _graph is not None:
        return _graph

    _checkpointer_context = AsyncPostgresSaver.from_conn_string(
        _checkpoint_uri(database_url or settings.DATABASE_URL)
    )
    _checkpointer = await _checkpointer_context.__aenter__()
    await _checkpointer.setup()
    _graph = build_interview_graph(checkpointer=_checkpointer)
    return _graph


async def get_graph() -> Any:
    """
    Return the initialized graph, lazily initializing in worker processes if needed.

    Returns:
        The compiled interview graph instance.
    """

    if _graph is None:
        return await init_graph()
    return _graph


async def close_graph() -> None:
    """
    Close the checkpointer connection owned by this process and flush background tasks.
    Must be called during application shutdown.
    """

    global _checkpointer, _checkpointer_context, _graph
    await drain_background_persistence()
    if _checkpointer_context is not None:
        await _checkpointer_context.__aexit__(None, None, None)
    _checkpointer = None
    _checkpointer_context = None
    _graph = None
