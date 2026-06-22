"""LangGraph assembly and invocation helpers for live interviews."""

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
    route_from_start,
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
from src.control.agents.nodes.persist import persist_turn
from src.control.agents.nodes.section_transition import section_transition
from src.control.agents.nodes.session_init import session_init
from src.control.agents.nodes.silence import handle_silence
from src.control.agents.nodes.skip import handle_skip
from src.control.agents.nodes.trigger_evaluation import trigger_evaluation
from src.control.agents.state import InterviewState


def build_graph():
    graph = StateGraph(InterviewState)

    graph.add_node("session_init", session_init)
    graph.add_node("opening", opening)
    graph.add_node("await_response", await_response)
    graph.add_node("classify_response", classify_response)
    graph.add_node("check_timers", check_timers)
    graph.add_node("evaluate_answer", evaluate_answer)
    graph.add_node("persist_turn", persist_turn)
    graph.add_node("generate_question", generate_question)
    graph.add_node("handle_clarification", handle_clarification)
    graph.add_node("handle_silence", handle_silence)
    graph.add_node("handle_irrelevant", handle_irrelevant)
    graph.add_node("handle_skip", handle_skip)
    graph.add_node("section_transition", section_transition)
    graph.add_node("closing", closing)
    graph.add_node("trigger_evaluation", trigger_evaluation)

    graph.add_conditional_edges(
        START,
        route_from_start,
        {
            "session_init": "session_init",
            "classify_response": "classify_response",
            "closing": "closing",
            END: END,
        },
    )

    graph.add_conditional_edges(
        "session_init",
        route_after_session_init,
        {"opening": "opening", "await_response": "await_response"},
    )
    graph.add_edge("opening", "await_response")
    graph.add_edge("await_response", END)

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

    graph.add_edge("evaluate_answer", "persist_turn")
    graph.add_edge("persist_turn", "generate_question")
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
    return graph.compile(checkpointer=MemorySaver())


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


async def invoke_graph(state_patch: dict, thread_id: str) -> dict:
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    return await graph.ainvoke(state_patch, config=config)


async def start_graph(*, candidate_assessment_id: str) -> dict:
    return await invoke_graph(
        {"candidate_assessment_id": candidate_assessment_id},
        thread_id=candidate_assessment_id,
    )


async def continue_graph(
    candidate_assessment_id: str,
    *,
    candidate_text: str,
    stt_confidence: float | None = None,
) -> dict:
    patch = {
        "candidate_raw_text": candidate_text,
        "candidate_stt_confidence": stt_confidence,
        "response_class": None,
        "bot_reply_text": "",
        "bot_reply_type": "",
        "next_node": None,
    }
    return await invoke_graph(patch, thread_id=candidate_assessment_id)


async def close_graph(
    candidate_assessment_id: str,
    *,
    terminated: bool = False,
    current_status: str = "COMPLETED",
) -> dict:
    patch = {
        "candidate_raw_text": "",
        "response_class": None,
        "bot_reply_text": "",
        "bot_reply_type": "",
        "should_close": True,
        "session_status": "TERMINATED" if terminated else current_status,
    }
    return await invoke_graph(patch, thread_id=candidate_assessment_id)
