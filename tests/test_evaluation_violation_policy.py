"""Deterministic holistic-evaluation policy for proctoring violations."""

from src.core.services.evaluation_score_calculator import (
    calculate_final_evaluation,
)
from src.schemas.evaluation_llm import (
    EvaluationCandidateContext,
    EvaluationInput,
    HolisticEvaluationLLMOutput,
    NvidiaEvaluationResult,
    QuestionAnswerPair,
    SeverityCounts,
    ViolationSummaryOutput,
)
from src.schemas.internal_evaluation import (
    EVALUATION_SCHEMA_VERSION,
    EvaluationContextBundle,
    TechnicalSkillSpec,
)
from src.utils.evaluation_llm import _semantic_errors
from src.utils.evaluation_violations import (
    deterministic_violation_evidence,
    normalize_evaluation_violations,
    singleton_policy_minimum_counts,
    violation_category_details,
)


def test_repeated_tab_switch_records_score_once_but_keep_exact_count() -> None:
    source = [
        {
            "violation_id": f"tab-switch:{index}",
            "violation_type": "tab_switch",
            "severity": "low",
            "timestamp": f"2026-07-19T10:00:0{index}Z",
            "metadata": {
                "event_id": str(index),
                "tab_switch_count": index,
            },
        }
        for index in range(1, 5)
    ]

    normalized = normalize_evaluation_violations(source)

    assert len(normalized) == 1
    violation = normalized[0]
    assert violation["severity"] == "low"
    assert violation["metadata"]["scored_occurrence_count"] == 1
    assert violation["metadata"]["occurrence_count"] == 4
    assert violation["metadata"]["tab_switch_count"] == 4
    assert len(violation["metadata"]["recorded_events"]) == 4
    assert singleton_policy_minimum_counts(normalized) == {
        "low": 1,
        "medium": 0,
        "high": 0,
        "critical": 0,
    }

    details = violation_category_details(normalized)
    assert details[0]["tab_switch_count"] == 4
    assert "detected 4 times" in deterministic_violation_evidence(normalized)[0]


def test_face_categories_are_high_score_once_and_preserve_durations() -> None:
    source = [
        {
            "violation_id": "face-absent:first",
            "violation_type": "face_absent",
            "severity": "low",
            "timestamp": "2026-07-19T10:01:00Z",
            "metadata": {
                "affects_evaluation": False,
                "occurrence_count": 1,
                "event_count": 2,
                "observed_duration_ms": 5_000,
                "max_face_count": 0,
            },
        },
        {
            "violation_id": "face-absent:legacy-repeat",
            "violation_type": "face_absent",
            "severity": "medium",
            "timestamp": "2026-07-19T10:02:00Z",
            "metadata": {
                "affects_evaluation": False,
                "observed_duration_ms": 30_000,
                "max_face_count": 0,
                "termination_triggered": True,
                "termination_reason": "face_absent_continuous_duration_exceeded",
                "termination_duration_ms": 30_000,
            },
        },
        {
            "violation_id": "multiple-faces:first",
            "violation_type": "multiple_faces",
            "severity": "medium",
            "timestamp": "2026-07-19T10:03:00Z",
            "metadata": {
                "affects_evaluation": False,
                "occurrence_count": 1,
                "episodes": [
                    {
                        "condition_started_at": "2026-07-19T10:02:40Z",
                        "max_observed_duration_ms": 20_000,
                    }
                ],
                "max_observed_duration_ms": 20_000,
                "max_face_count": 3,
                "termination_triggered": True,
                "termination_reason": ("multiple_faces_continuous_duration_exceeded"),
                "termination_duration_ms": 20_000,
            },
        },
    ]

    normalized = normalize_evaluation_violations(source)

    assert len(normalized) == 2
    assert [item["violation_type"] for item in normalized] == [
        "face_absent",
        "multiple_faces",
    ]
    assert all(item["severity"] == "high" for item in normalized)
    assert singleton_policy_minimum_counts(normalized) == {
        "low": 0,
        "medium": 0,
        "high": 2,
        "critical": 0,
    }

    absent_metadata = normalized[0]["metadata"]
    assert absent_metadata["scored_occurrence_count"] == 1
    assert absent_metadata["occurrence_count"] == 2
    assert absent_metadata["observed_durations_ms"] == [5_000, 30_000]
    assert absent_metadata["max_observed_duration_ms"] == 30_000
    assert absent_metadata["total_observed_duration_ms"] == 35_000
    assert absent_metadata["termination_triggered"] is True
    assert absent_metadata["termination_reason"] == (
        "face_absent_continuous_duration_exceeded"
    )
    assert absent_metadata["termination_duration_ms"] == 30_000

    details = {
        item["violation_type"]: item for item in violation_category_details(normalized)
    }
    assert details["face_absent"]["max_observed_duration_ms"] == 30_000
    assert details["multiple_faces"]["max_observed_duration_ms"] == 20_000
    assert details["multiple_faces"]["observed_durations_ms"] == [20_000]
    assert details["multiple_faces"]["termination_triggered"] is True
    assert details["multiple_faces"]["termination_duration_ms"] == 20_000
    evidence = deterministic_violation_evidence(normalized)
    assert any("30.0 seconds" in item for item in evidence)
    assert any("20.0 seconds" in item for item in evidence)


def test_face_transport_updates_do_not_inflate_episode_occurrences() -> None:
    normalized = normalize_evaluation_violations(
        [
            {
                "violation_id": "proctoring:face_absent",
                "violation_type": "face_absent",
                "severity": "high",
                "metadata": {
                    "occurrence_count": 1,
                    "event_count": 2,
                    "episodes": [
                        {
                            "condition_started_at": "2026-07-19T10:00:00Z",
                            "max_observed_duration_ms": 30_000,
                        }
                    ],
                },
            }
        ]
    )

    assert normalized[0]["metadata"]["occurrence_count"] == 1
    assert violation_category_details(normalized)[0]["occurrence_count"] == 1


def test_non_policy_records_keep_existing_review_only_behavior() -> None:
    normalized = normalize_evaluation_violations(
        [
            {
                "violation_id": "keep",
                "violation_type": "prompt_injection",
                "severity": "critical",
                "metadata": {"affects_evaluation": True},
            },
            {
                "violation_id": "drop",
                "violation_type": "experimental_signal",
                "severity": "low",
                "metadata": {"affects_evaluation": False},
            },
        ]
    )

    assert [item["violation_id"] for item in normalized] == ["keep"]


def _evaluation_bundle(
    violations: list[dict[str, object]],
) -> EvaluationContextBundle:
    question = QuestionAnswerPair(
        question_id="q-1",
        section="technical",
        skill="Python",
        difficulty="medium",
        question_text="How would you design a reliable worker?",
        answers=["I would use idempotency and retry limits."],
        response_types=["answer"],
        answered=True,
    )
    evaluation_input = EvaluationInput(
        evaluation_schema_version=EVALUATION_SCHEMA_VERSION,
        transcript_hash="a" * 64,
        candidate=EvaluationCandidateContext(
            candidate_assessment_id="candidate-1",
            session_id="session-1",
            assessment_id="assessment-1",
            candidate_name="Candidate",
            assessment_title="Assessment",
            role_name="Engineer",
            company_name="Company",
            inferred_difficulty="mid-level",
            interview_duration_mins=30,
            total_elapsed_secs=600,
        ),
        jd_analysis={},
        interview_plan={},
        transcript=[{"speaker": "bot", "text": question.question_text}],
        qa_pairs=[question],
        violations=violations,
    )
    return EvaluationContextBundle(
        evaluation_input=evaluation_input,
        technical_skills=(
            TechnicalSkillSpec(
                name="Python",
                priority_score=8.0,
                expected_signals=(),
                questions_asked=1,
            ),
        ),
    )


def _model_result() -> NvidiaEvaluationResult:
    raw_output = {
        "intro_section_score": 6.0,
        "intro_section_summary": "The introduction was adequate.",
        "intro_section_evidence": [],
        "skill_scores": {
            "Python": {
                "score": 7.0,
                "priority_score": 8.0,
                "questions_evaluated": 1,
                "confidence": 0.8,
            }
        },
        "skill_summary": {"Python": "The answer covered reliability."},
        "skill_evidence": {"Python": ["idempotency and retry limits"]},
        "overall_technical_skill_score": 7.0,
        "question_evaluations": [
            {
                "question_id": "q-1",
                "section": "technical",
                "skill": "Python",
                "difficulty": "medium",
                "question_text": "How would you design a reliable worker?",
                "answered": True,
                "answer_summary": "The candidate proposed reliability controls.",
                "score": 7.0,
                "relevance_class": "direct_match",
                "evidence": ["idempotency and retry limits"],
                "confidence": 0.8,
            }
        ],
        "behavioural_cultural_score": 6.0,
        "behavioural_cultural_summary": "Limited behavioural evidence.",
        "behavioural_cultural_evidence": ["No behavioural question was asked."],
        "communication_score": 7.0,
        "communication_summary": "The answer was clear.",
        "communication_evidence": ["A concise direct answer."],
        "section_communication_scores": {
            section: {
                "score": 6.0,
                "summary": "Limited but understandable evidence.",
                "evidence": [],
            }
            for section in ("self_intro", "technical", "behavioural_cultural")
        },
        "violation_summary": {
            "has_violation": True,
            "validated_violation_count": 3,
            "severity_counts": {
                "low": 1,
                "medium": 0,
                "high": 2,
                "critical": 0,
            },
            "summary": "One tab and two face categories were recorded.",
        },
        "violation_evidence": ["Automated proctoring records were supplied."],
        "overall_score": 6.5,
        "hiring_recommendation": "consider",
        "overall_summary": "The interview provided sufficient evidence.",
        "recommendation_reasoning": "Technical evidence was adequate.",
        "strengths": ["Python"],
        "concerns": [],
    }
    return NvidiaEvaluationResult(
        output=HolisticEvaluationLLMOutput.model_validate(raw_output),
        raw_output=raw_output,
    )


def test_final_record_persists_exact_policy_details_and_scores_categories_once() -> (
    None
):
    normalized = normalize_evaluation_violations(
        [
            {
                "violation_id": "tab",
                "violation_type": "tab_switch",
                "severity": "low",
                "metadata": {"tab_switch_count": 4},
            },
            {
                "violation_id": "absent",
                "violation_type": "face_absent",
                "severity": "high",
                "metadata": {
                    "observed_duration_ms": 30_000,
                    "max_face_count": 0,
                },
            },
            {
                "violation_id": "multiple",
                "violation_type": "multiple_faces",
                "severity": "high",
                "metadata": {
                    "observed_duration_ms": 20_000,
                    "max_face_count": 2,
                },
            },
        ]
    )

    record = calculate_final_evaluation(
        bundle=_evaluation_bundle(normalized),
        model_result=_model_result(),
    )

    assert record.violation_penalty == 0.45
    assert record.violation_summary["severity_counts"] == {
        "low": 1,
        "medium": 0,
        "high": 2,
        "critical": 0,
    }
    details = {
        item["violation_type"]: item
        for item in record.violation_summary["category_details"]
    }
    assert details["tab_switch"]["tab_switch_count"] == 4
    assert details["face_absent"]["max_observed_duration_ms"] == 30_000
    assert details["multiple_faces"]["max_observed_duration_ms"] == 20_000
    assert any("detected 4 times" in item for item in record.violation_evidence)
    assert any("30.0 seconds" in item for item in record.violation_evidence)
    assert any("20.0 seconds" in item for item in record.violation_evidence)


def test_evaluator_cannot_omit_or_downgrade_authoritative_face_categories() -> None:
    violations = normalize_evaluation_violations(
        [
            {
                "violation_type": "face_absent",
                "severity": "high",
                "metadata": {"observed_duration_ms": 30_000},
            },
            {
                "violation_type": "multiple_faces",
                "severity": "high",
                "metadata": {
                    "observed_duration_ms": 20_000,
                    "max_face_count": 2,
                },
            },
        ]
    )
    output = _model_result().output.model_copy(
        update={
            "violation_summary": ViolationSummaryOutput(
                has_violation=True,
                validated_violation_count=2,
                severity_counts=SeverityCounts(
                    low=0,
                    medium=2,
                    high=0,
                    critical=0,
                ),
                summary="The face records were incorrectly downgraded.",
            )
        }
    )

    errors = _semantic_errors(output, _evaluation_bundle(violations))

    assert any("severity_counts.high must be at least 2" in item for item in errors)
