"""JD question-brief propagation into both live generation paths."""

import json

from src.control.agents.prompts import LIVE_INTERVIEWER_SYSTEM_PROMPT
from src.control.agents.utils.generate_question import _technical_messages
from src.control.agents.utils.initialize_context import _runtime_sections
from src.control.agents.utils.interviewer_turn import _live_interviewer_messages
from src.schemas.prompts import CandidateResponseClassification


def _plan(*, include_brief: bool = True) -> dict[str, object]:
    technical: dict[str, object] = {
        "section_name": "Python",
        "skill": "Python",
        "allocated_mins": 8.0,
        "expected_signals": ["Dependency injection", "Failure handling"],
    }
    if include_brief:
        technical["question_brief"] = {
            "expected_signals": ["Dependency injection", "Failure handling"],
            "role_responsibility": "Build reliable Python APIs.",
            "operating_environment": "FastAPI services in production.",
            "important_tools": ["Python", "FastAPI"],
            "constraints": ["Latency", "Reliability"],
            "seniority_depth": "Use mid-level production trade-offs.",
            "out_of_scope_topics": ["Desktop GUI development"],
        }
    return {
        "total_mins": 10,
        "inferred_difficulty": "mid-level",
        "sections": [technical],
    }


def _state(section: dict[str, object]) -> dict[str, object]:
    return {
        "runtime_sections": [section],
        "current_section_index": 0,
        "current_section_kind": "technical",
        "current_technical_skill": "Python",
        "current_question_text": "How does dependency injection help an API?",
        "current_question_difficulty": "medium",
        "previous_candidate_response": "It separates dependencies for testing.",
        "inferred_difficulty": "mid-level",
        "resume_context": {},
        "asked_questions": [],
        "used_topics_by_skill": {},
        "skill_evaluation_streaks": {},
        "question_variation_seed": "candidate-seed",
    }


def test_brief_reaches_integrated_and_fallback_question_prompts() -> None:
    section = _runtime_sections(_plan())[0]
    state = _state(section)
    target = {
        "kind": "technical",
        "skill": "Python",
        "signals": section["expected_signals"],
        "question_brief": section["question_brief"],
        "entering_new_section": False,
    }
    classification = CandidateResponseClassification(
        response_type="answer",
        clarification_type=None,
        is_substantial=True,
        interview_meta_type=None,
    )

    integrated = _live_interviewer_messages(
        state,  # type: ignore[arg-type]
        classification=classification,
        target=target,
        decision={"action": "stay"},
        ask_next_question=True,
        must_close=False,
        plan={
            strength: {
                "difficulty": "medium",
                "probe_deeper": False,
                "follow_interesting_thread": False,
            }
            for strength in ("weak", "adequate", "strong")
        },
    )
    fallback = _technical_messages(
        state,  # type: ignore[arg-type]
        skill="Python",
        target_difficulty="medium",
        probe_deeper=False,
    )

    integrated_context = json.loads(integrated[1]["content"])
    fallback_context = json.loads(fallback[1]["content"])
    assert integrated_context["jd_question_brief"] == section["question_brief"]
    assert fallback_context["jd_question_brief"] == section["question_brief"]
    assert "gentle steering context" in LIVE_INTERVIEWER_SYSTEM_PROMPT


def test_legacy_plan_gets_bounded_role_grounded_fallback() -> None:
    sections = _runtime_sections(
        _plan(include_brief=False),
        role_name="Backend Engineer",
        jd_analysis={
            "skills": [
                {
                    "skill": "Python",
                    "priority_score": 9.0,
                    "reasoning": "Python is used to build backend APIs.",
                },
                {
                    "skill": "Desktop GUI",
                    "priority_score": 2.0,
                    "reasoning": "Optional exposure only.",
                },
            ]
        },
    )

    brief = sections[0]["question_brief"]
    assert brief["role_responsibility"] == "Python is used to build backend APIs."
    assert brief["seniority_depth"].startswith("Use mid-level")
    assert brief["out_of_scope_topics"] == ["Desktop GUI"]
