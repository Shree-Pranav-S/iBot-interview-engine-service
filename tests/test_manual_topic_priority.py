"""Evaluation priorities for recruiter-edited interview plans."""

import pytest

from src.utils.evaluation_context import _technical_skill_specs


def test_jd_priority_remains_authoritative_for_matching_topic() -> None:
    specs = _technical_skill_specs(
        jd_analysis={
            "skills": [{"skill": "Python", "priority_score": 9.0}],
        },
        interview_plan={
            "sections": [
                {
                    "section_name": "Python",
                    "skill": "Python",
                    "allocated_mins": 2.0,
                },
                {
                    "section_name": "System Design",
                    "skill": "System Design",
                    "allocated_mins": 8.0,
                },
            ]
        },
    )

    assert specs[0].priority_score == 9.0


def test_manual_topic_priority_scales_around_average_allocated_time() -> None:
    specs = _technical_skill_specs(
        jd_analysis={"skills": []},
        interview_plan={
            "sections": [
                {
                    "section_name": "API Design",
                    "skill": "API Design",
                    "allocated_mins": 2.0,
                },
                {
                    "section_name": "Distributed Systems",
                    "skill": "Distributed Systems",
                    "allocated_mins": 8.0,
                },
            ]
        },
    )

    assert specs[0].priority_score == pytest.approx(2.0)
    assert specs[1].priority_score == pytest.approx(8.0)


def test_manual_topic_without_timing_keeps_legacy_fallback() -> None:
    specs = _technical_skill_specs(
        jd_analysis={"skills": []},
        interview_plan={
            "sections": [
                {
                    "section_name": "API Design",
                    "skill": "API Design",
                }
            ]
        },
    )

    assert specs[0].priority_score == 1.0
