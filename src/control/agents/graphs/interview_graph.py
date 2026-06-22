"""LangGraph assembly for live interviews."""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.control.agents.graphs.edges import (
    route_after_generate,
    route_after_irrelevant,
    route_after_session_init,
    route_after_silence,
    route_after_skip,
    route_after_timer_check,
)
from src.control.agents.nodes.await_response import await_response
from src.control.agents.nodes.check_timers import check_timers
from src.control.agents.nodes.clarify import handle_clarification
from src.control.agents.nodes.classify import classify_response
from src.control.agents.nodes.closing import closing
from src.control.agents.nodes.evaluate import evaluate_answer
from src.control.agents.nodes.generate_question import generate_question
from src.control.agents.nodes.irrelevant import handle_irrelevant
from src.control.agents.nodes.opening import opening
from src.control.agents.nodes.section_transition import section_transition
from src.control.agents.nodes.session_init import session_init
from src.control.agents.nodes.silence import handle_silence
from src.control.agents.nodes.skip import handle_skip
from src.control.agents.nodes.trigger_evaluation import trigger_evaluation
from src.control.agents.state import InterviewState

_checkpointer = MemorySaver()


def build_graph():
    graph = StateGraph(InterviewState)

    graph.add_node("session_init", session_init)
    graph.add_node("opening", opening)
    graph.add_node("await_response", await_response)
    graph.add_node("classify_response", classify_response)
    graph.add_node("check_timers", check_timers)
    graph.add_node("evaluate_answer", evaluate_answer)
    graph.add_node("generate_question", generate_question)
    graph.add_node("handle_clarification", handle_clarification)
    graph.add_node("handle_silence", handle_silence)
    graph.add_node("handle_irrelevant", handle_irrelevant)
    graph.add_node("handle_skip", handle_skip)
    graph.add_node("section_transition", section_transition)
    graph.add_node("closing", closing)
    graph.add_node("trigger_evaluation", trigger_evaluation)

    graph.add_edge(START, "session_init")
    graph.add_conditional_edges(
        "session_init",
        route_after_session_init,
        {"opening": "opening", "await_response": "await_response"},
    )
    graph.add_edge("opening", "await_response")

    # On first entry this node interrupts. On Command(resume=...), it returns
    # the transcript patch and the graph continues through the interview turn.
    graph.add_edge("await_response", "classify_response")

    graph.add_edge("classify_response", "check_timers")
    graph.add_conditional_edges(
        "check_timers",
        route_after_timer_check,
        {
            "evaluate_answer": "evaluate_answer",
            "handle_clarification": "handle_clarification",
            "handle_silence": "handle_silence",
            "handle_irrelevant": "handle_irrelevant",
            "handle_skip": "handle_skip",
            "section_transition": "section_transition",
            "closing": "closing",
        },
    )

    graph.add_edge("evaluate_answer", "generate_question")
    graph.add_conditional_edges(
        "generate_question",
        route_after_generate,
        {
            "section_transition": "section_transition",
            "closing": "closing",
            "await_response": "await_response",
        },
    )

    graph.add_edge("handle_clarification", "await_response")
    graph.add_conditional_edges(
        "handle_silence",
        route_after_silence,
        {
            "generate_question": "generate_question",
            "closing": "closing",
            "await_response": "await_response",
        },
    )
    graph.add_conditional_edges(
        "handle_irrelevant",
        route_after_irrelevant,
        {"closing": "closing", "await_response": "await_response"},
    )
    graph.add_conditional_edges(
        "handle_skip",
        route_after_skip,
        {"generate_question": "generate_question", "await_response": "await_response"},
    )

    graph.add_conditional_edges(
        "section_transition",
        lambda state: state.get("next_node", "generate_question"),
        {
            "generate_question": "generate_question",
            "closing": "closing",
        },
    )
    graph.add_edge("closing", "trigger_evaluation")
    graph.add_edge("trigger_evaluation", END)
    return graph.compile(checkpointer=_checkpointer)


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()
