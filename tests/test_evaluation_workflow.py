"""Focused coverage for the structured holistic evaluation workflow."""

from __future__ import annotations

from typing import Any

import pytest

from src.config.settings import settings
from src.core.services.evaluation_context_builder import build_evaluation_context
from src.core.services.evaluation_llm_client import (
    NvidiaEvaluationResult,
    _base_messages,
    _requires_two_stage,
    _semantic_errors,
)
from src.core.services.evaluation_prompt import HOLISTIC_EVALUATION_SYSTEM_PROMPT
from src.core.services.evaluation_score_calculator import calculate_final_evaluation
from src.schemas.evaluation_llm import HolisticEvaluationLLMOutput


def _source(*, include_behavioural_answer: bool = True) -> dict[str, Any]:
    transcript: list[dict[str, Any]] = [
        {
            "speaker": "bot",
            "turn_id": "bot-1",
            "question_id": "intro-1",
            "question_type": "opening",
            "current_section": "self_intro",
            "text": "Please introduce yourself.",
        },
        {
            "speaker": "candidate",
            "turn_id": "candidate-1",
            "question_id": "intro-1",
            "response_type": "answer",
            "text": "I build Python APIs and enjoy improving backend reliability.",
        },
        {
            "speaker": "bot",
            "turn_id": "bot-2",
            "question_id": "python-1",
            "question_type": "new_question",
            "current_section": "Python",
            "current_skill": "python",
            "question_difficulty": "medium",
            "text": "When would you use a Python generator?",
            "metadata": {"expected_signals": ["lazy iteration", "memory efficiency"]},
        },
        {
            "speaker": "candidate",
            "turn_id": "candidate-2",
            "question_id": "python-1",
            "text": "A generator yields values lazily.",
            "metadata": {
                "classification": {"response_type": "answer"},
                "live_evaluation": {"answer_strength": "partial"},
            },
        },
        {
            "speaker": "bot",
            "turn_id": "bot-3",
            "question_id": "python-1",
            "question_type": "clarification_response",
            "current_section": "Python",
            "current_skill": "Python",
            "text": "Can you explain the memory benefit?",
        },
        {
            "speaker": "candidate",
            "turn_id": "candidate-3",
            "question_id": "python-1",
            "response_type": "answer",
            "text": "It avoids materializing the entire collection in memory.",
        },
        {
            "speaker": "bot",
            "turn_id": "bot-4",
            "question_id": "behaviour-1",
            "question_type": "new_question",
            "current_section": "behavioural_cultural",
            "text": "How do you handle feedback?",
        },
    ]
    if include_behavioural_answer:
        transcript.append(
            {
                "speaker": "candidate",
                "turn_id": "candidate-4",
                "question_id": "behaviour-1",
                "response_type": "answer",
                "text": "I listen, clarify, and act on it.",
            }
        )

    return {
        "candidate_assessment_id": "00000000-0000-0000-0000-000000000001",
        "session_id": "00000000-0000-0000-0000-000000000002",
        "assessment_id": "00000000-0000-0000-0000-000000000003",
        "candidate_name": "Candidate",
        "assessment_title": "Backend interview",
        "role_name": "Backend Engineer",
        "interview_duration_mins": 30,
        "total_elapsed_secs": 1200,
        "jd_analysis": {
            "skills": [{"skill": "Python", "priority_score": 8.0}],
        },
        "interview_plan": {
            "inferred_difficulty": "junior",
            "sections": [
                {"section_name": "self_intro"},
                {
                    "section_name": "Python",
                    "skill": "Python",
                    "expected_signals": ["lazy iteration", "memory efficiency"],
                },
                {"section_name": "behavioural_cultural"},
            ],
        },
        "transcript": transcript,
        "violations": [],
    }


def _model_output(
    bundle: Any,
    *,
    technical_question_score: float = 9.0,
    technical_relevance: str = "close_equivalent",
    behavioural_score: float = 6.0,
) -> HolisticEvaluationLLMOutput:
    question_evaluations = []
    for pair in bundle.evaluation_input.qa_pairs:
        is_technical = pair.skill == "Python"
        question_evaluations.append(
            {
                "question_id": pair.question_id,
                "section": pair.section,
                "skill": pair.skill,
                "difficulty": pair.difficulty,
                "question_text": pair.question_text,
                "answered": pair.answered,
                "answer_summary": (
                    "The candidate gave a substantive answer."
                    if pair.answered
                    else "No substantive answer was provided."
                ),
                "score": technical_question_score if is_technical else 6.0,
                "relevance_class": (
                    technical_relevance if is_technical else "direct_match"
                ),
                "evidence": ["Evidence grounded in the paired candidate response."],
                "confidence": 0.8,
            }
        )

    return HolisticEvaluationLLMOutput.model_validate(
        {
            "intro_section_score": 6.0,
            "intro_section_summary": "A concise, role-relevant introduction.",
            "intro_section_evidence": ["The candidate described Python API work."],
            "skill_scores": {
                "Python": {
                    "score": 3.0,
                    "priority_score": 8.0,
                    "questions_evaluated": 1,
                    "confidence": 0.4,
                }
            },
            "skill_summary": {"Python": "The paired answer shows useful knowledge."},
            "skill_evidence": {"Python": ["The generator answer explained laziness."]},
            "overall_technical_skill_score": 3.0,
            "question_evaluations": question_evaluations,
            "behavioural_cultural_score": behavioural_score,
            "behavioural_cultural_summary": "Limited but professional evidence.",
            "behavioural_cultural_evidence": ["The candidate engaged professionally."],
            "communication_score": 6.0,
            "communication_summary": "The candidate communicated clearly.",
            "communication_evidence": ["Answers were understandable."],
            "section_communication_scores": {
                "self_intro": {
                    "score": 6.0,
                    "summary": "Clear introduction.",
                    "evidence": ["Relevant background was stated."],
                },
                "technical": {
                    "score": 6.0,
                    "summary": "Technical meaning was clear.",
                    "evidence": ["The explanation was concise."],
                },
                "behavioural_cultural": {
                    "score": 6.0,
                    "summary": "Professional but brief.",
                    "evidence": ["The response was constructive."],
                },
            },
            "violation_summary": {
                "has_violation": False,
                "validated_violation_count": 0,
                "severity_counts": {
                    "low": 0,
                    "medium": 0,
                    "high": 0,
                    "critical": 0,
                },
                "summary": "No validated violations.",
            },
            "violation_evidence": [],
            "overall_score": 6.0,
            "hiring_recommendation": "consider",
            "overall_summary": "The candidate showed a useful foundation.",
            "recommendation_reasoning": "Technical evidence supports consideration.",
            "strengths": [],
            "concerns": [],
        }
    )


def test_context_builder_pairs_followups_and_canonicalizes_skills() -> None:
    bundle = build_evaluation_context(_source())

    assert bundle.evaluation_input.evaluation_schema_version == "holistic-evaluation-v2"
    assert [item.question_id for item in bundle.evaluation_input.qa_pairs] == [
        "intro-1",
        "python-1",
        "behaviour-1",
    ]
    technical_pair = bundle.evaluation_input.qa_pairs[1]
    assert technical_pair.skill == "Python"
    assert technical_pair.answers == [
        "A generator yields values lazily.",
        "It avoids materializing the entire collection in memory.",
    ]
    assert technical_pair.response_types == ["answer", "answer"]
    assert technical_pair.answered is True
    assert bundle.technical_skills[0].questions_asked == 1


def test_scorer_uses_question_scores_and_enforces_relevance_cap() -> None:
    bundle = build_evaluation_context(_source())
    output = _model_output(bundle)

    record = calculate_final_evaluation(
        bundle=bundle,
        model_result=NvidiaEvaluationResult(
            output=output,
            raw_output={"pipeline": "direct_qa"},
        ),
    )

    assert record.skill_scores["Python"]["score"] == 8.0
    assert record.overall_technical_skill_score == 8.0
    assert record.question_evaluations[1]["score"] == 8.0
    assert record.evaluation_schema_version == "holistic-evaluation-v2"


def test_scoring_payload_omits_raw_transcript_and_long_threshold_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = build_evaluation_context(_source())
    message = str(_base_messages(bundle)[1]["content"])

    assert '"qa_pairs":' in message
    assert '"transcript":' not in message

    monkeypatch.setattr(settings, "NVIDIA_NIM_TWO_STAGE_QA_THRESHOLD", 3)
    monkeypatch.setattr(settings, "NVIDIA_NIM_TWO_STAGE_INPUT_CHARS", 999999)
    assert _requires_two_stage(bundle) is True


def test_semantic_validation_requires_exact_question_coverage() -> None:
    bundle = build_evaluation_context(_source())
    output = _model_output(bundle)
    output.question_evaluations.pop()

    errors = _semantic_errors(output, bundle)

    assert any("question_evaluations" in error for error in errors)


def test_prompt_is_focused_and_contains_behavioural_leniency() -> None:
    assert "- the answe" not in HOLISTIC_EVALUATION_SYSTEM_PROMPT
    assert "brevity alone" in HOLISTIC_EVALUATION_SYSTEM_PROMPT.casefold()
    assert "one question_evaluations entry" in HOLISTIC_EVALUATION_SYSTEM_PROMPT
