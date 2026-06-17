"""
Interview Graph — LangGraph StateGraph for the interview workflow.

Hub-and-spoke topology with classify_response as the central router.
All nodes checkpoint to Postgres via AsyncPostgresSaver.

Graph topology:
    START → init_session → deliver_opening → await_response
    → classify_response → [route_response] →
        answer:        evaluate_answer → [section check] →
                            generate_question → await_response
                            section_transition → [complete check] →
                                generate_question | interview_complete
        clarification: handle_clarification → await_response
        silence:       handle_silence → [silence check] →
                            await_response | generate_question
        irrelevant:    handle_irrelevant → [complete check] →
                            generate_question | interview_complete
    interview_complete → trigger_evaluation → END
"""

from __future__ import annotations

import logging

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, StateGraph

from src.control.agents.nodes.await_response import await_response_node
from src.control.agents.nodes.classify_response import (
    classify_response_node,
    route_response,
)
from src.control.agents.nodes.deliver_opening import deliver_opening_node
from src.control.agents.nodes.evaluate_answer import evaluate_answer_node
from src.control.agents.nodes.generate_question import generate_question_node
from src.control.agents.nodes.handle_clarification import handle_clarification_node
from src.control.agents.nodes.handle_irrelevant import handle_irrelevant_node
from src.control.agents.nodes.handle_silence import handle_silence_node
from src.control.agents.nodes.init_session import init_session_node
from src.control.agents.nodes.interview_complete import interview_complete_node
from src.control.agents.nodes.section_transition import section_transition_node
from src.control.agents.nodes.trigger_evaluation import trigger_evaluation_node
from src.control.agents.state import InterviewState

logger = logging.getLogger(__name__)


# ── Conditional edge functions ────────────────────────────────────────────────


def _should_complete(state: InterviewState) -> str:
    """Route after section_transition or handle_irrelevant: complete or continue."""
    status = state.get("session_status", "in_progress")
    if status in ("completed", "terminated", "deactivated"):
        return "interview_complete"
    return "generate_question"


def _after_silence(state: InterviewState) -> str:
    """Route after handle_silence: if third attempt gave zero score, move on."""
    silence_attempt = state.get("silence_attempt", 0)
    if silence_attempt == 0:
        # Was reset to 0 → zero-score path, generate new question
        return "generate_question"
    # Still in silence protocol → wait for response again
    return "await_response"


def _after_evaluation(state: InterviewState) -> str:
    """
    Route after evaluate_answer: check if the section should transition.

    A section transition is triggered when:
    1. Section time remaining ≤ 0 AND at least _MIN_QUESTIONS_PER_SECTION
       have been asked in this section.
    2. The session status is still in_progress.

    Otherwise, continue to generate_question.
    """
    status = state.get("session_status", "in_progress")
    if status != "in_progress":
        return "section_transition"

    time_remaining = state.get("current_section_time_remaining_secs", 300)
    sections = state.get("sections", [])
    section_idx = state.get("current_section_index", 0)

    current_section = sections[section_idx] if section_idx < len(sections) else None
    questions_asked = (
        current_section.get("questions_asked", 0) if current_section else 0
    )

    if time_remaining <= 0:
        logger.info(
            "Section time expired (remaining=%ds, questions=%d) — transitioning",
            time_remaining,
            questions_asked,
        )
        return "section_transition"

    return "generate_question"


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_interview_graph() -> StateGraph:
    """
    Build the full interview StateGraph with hub-and-spoke topology.

    Returns the uncompiled graph — call compile() with a checkpointer
    to get an executable graph.
    """
    graph = StateGraph(InterviewState)

    # ── Register all nodes ────────────────────────────────────────────────────

    graph.add_node("init_session", init_session_node)
    graph.add_node("deliver_opening", deliver_opening_node)
    graph.add_node("await_response", await_response_node)
    graph.add_node("classify_response", classify_response_node)
    graph.add_node("evaluate_answer", evaluate_answer_node)
    graph.add_node("generate_question", generate_question_node)
    graph.add_node("handle_clarification", handle_clarification_node)
    graph.add_node("handle_silence", handle_silence_node)
    graph.add_node("handle_irrelevant", handle_irrelevant_node)
    graph.add_node("section_transition", section_transition_node)
    graph.add_node("interview_complete", interview_complete_node)
    graph.add_node("trigger_evaluation", trigger_evaluation_node)

    # ── Wire the edges ────────────────────────────────────────────────────────

    # Linear entry path
    graph.set_entry_point("init_session")
    graph.add_edge("init_session", "deliver_opening")
    graph.add_edge("deliver_opening", "await_response")

    # await_response → classify_response (always)
    graph.add_edge("await_response", "classify_response")

    # classify_response → [conditional hub dispatch]
    graph.add_conditional_edges(
        "classify_response",
        route_response,
        {
            "answer": "evaluate_answer",
            "clarification": "handle_clarification",
            "silence": "handle_silence",
            "irrelevant": "handle_irrelevant",
            "time_up": "section_transition",
        },
    )

    # evaluate_answer → [conditional: section transition or next question]
    graph.add_conditional_edges(
        "evaluate_answer",
        _after_evaluation,
        {
            "generate_question": "generate_question",
            "section_transition": "section_transition",
        },
    )

    # generate_question → await_response (loop back for next turn)
    graph.add_edge("generate_question", "await_response")

    # handle_clarification → await_response (same question, wait again)
    graph.add_edge("handle_clarification", "await_response")

    # handle_silence → conditional: continue silence protocol or move on
    graph.add_conditional_edges(
        "handle_silence",
        _after_silence,
        {
            "await_response": "await_response",
            "generate_question": "generate_question",
        },
    )

    # handle_irrelevant → conditional: continue or terminate
    graph.add_conditional_edges(
        "handle_irrelevant",
        _should_complete,
        {
            "interview_complete": "interview_complete",
            "generate_question": "generate_question",
        },
    )

    # section_transition → conditional: complete or next question
    graph.add_conditional_edges(
        "section_transition",
        _should_complete,
        {
            "interview_complete": "interview_complete",
            "generate_question": "generate_question",
        },
    )

    # Terminal path
    graph.add_edge("interview_complete", "trigger_evaluation")
    graph.add_edge("trigger_evaluation", END)

    logger.info("Interview graph built successfully")
    return graph


# ── Graph compiler ────────────────────────────────────────────────────────────


async def compile_interview_graph(
    db_uri: str,
) -> tuple:
    """
    Compile the interview graph with Postgres checkpointer.

    Args:
        db_uri: PostgreSQL connection string (psycopg format).

    Returns:
        (compiled_graph, checkpointer_context) tuple.
    """
    graph = build_interview_graph()

    checkpointer_context = AsyncPostgresSaver.from_conn_string(db_uri)
    checkpointer = await checkpointer_context.__aenter__()
    await checkpointer.setup()

    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("Interview graph compiled with Postgres checkpointer")
    return compiled, checkpointer_context
