"""Final holistic interview evaluation service."""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from src.core.services.llm_service import evaluate
from src.data.repositories.interview_repository import (
    get_interview_data_for_evaluation,
    save_holistic_evaluation,
)
from src.schemas.evaluation import (
    BehaviouralCulturalEvaluation,
    FinalSynthesisReport,
    HolisticEvaluationReport,
    IntroSectionEvaluation,
    SectionSummaryReport,
    SkillScoreReport,
)
from src.utils.evaluation import (
    _build_model_audit_context,
    _candidate_answer_records,
    _is_candidate_turn,
    _is_technical_section,
)
from src.utils.holistic_evaluation_prompt import (
    BEHAVIOURAL_CULTURAL_SYSTEM_PROMPT,
    FINAL_SYNTHESIS_SYSTEM_PROMPT,
    INTRO_SECTION_SYSTEM_PROMPT,
    TECHNICAL_SKILL_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

EVIDENCE_PACK_MAX_CHARS = 9000
REPAIR_RESPONSE_MAX_CHARS = 6000
_JSON_SEPARATORS = (",", ":")
STAGE_EVIDENCE_MAX_CHARS = 7200
FINAL_SYNTHESIS_MAX_CHARS = 18000

SELF_INTRO_SECTION_KEYS = {"self_intro", "intro", "introduction"}
BEHAVIOURAL_CULTURAL_SECTION_KEYS = {
    "behavioural",
    "behavioral",
    "cultural",
    "culture",
    "behavioural_cultural",
    "behavioral_cultural",
}
ModelT = TypeVar("ModelT", bound=BaseModel)


TECHNICAL_SKILL_KEYS = {
    "skills",
    "technical_skills",
    "programming_languages",
    "frameworks",
    "tools",
    "technologies",
    "databases",
    "cloud",
    "languages",
}


def _parse_model_report(raw: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Holistic evaluation response must be a JSON object")
    report = HolisticEvaluationReport.model_validate(parsed)
    return report.model_dump(mode="json")


def _parse_model_payload(raw: str, model: type[ModelT]) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Evaluation response must be a JSON object")
    return model.model_validate(parsed).model_dump(mode="json")


def _json_blob(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=_JSON_SEPARATORS,
        default=str,
    )


def _truncate_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _compact_text_list(
    value: Any,
    *,
    max_items: int = 8,
    max_chars: int = 140,
) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            raw = item.get("skill") or item.get("name") or item.get("title")
        else:
            raw = item
        text = _truncate_text(raw, max_chars)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            cleaned.append(text)
        if len(cleaned) >= max_items:
            break
    return cleaned


def _compact_json_value(value: Any, max_chars: int = 1600) -> Any:
    blob = _json_blob(value)
    if len(blob) <= max_chars:
        return value
    return _truncate_text(blob, max_chars)


def _collect_resume_skills(value: Any, output: list[str], *, depth: int = 0) -> None:
    if depth > 4 or len(output) >= 40:
        return

    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key or "").lower()
            if any(skill_key in normalized for skill_key in TECHNICAL_SKILL_KEYS):
                output.extend(_compact_text_list(item, max_items=40, max_chars=80))
            elif isinstance(item, dict | list):
                _collect_resume_skills(item, output, depth=depth + 1)
        return

    if isinstance(value, list):
        for item in value:
            if len(output) >= 40:
                return
            if isinstance(item, dict):
                raw = item.get("skill") or item.get("name") or item.get("title")
                if raw:
                    output.append(_truncate_text(raw, 80))
                _collect_resume_skills(item, output, depth=depth + 1)
            elif isinstance(item, str):
                output.append(_truncate_text(item, 80))


def _resume_skill_names(resume_parsed: Any) -> list[str]:
    skills: list[str] = []
    _collect_resume_skills(resume_parsed, skills)
    cleaned: list[str] = []
    seen: set[str] = set()
    for skill in skills:
        key = skill.lower()
        if skill and key not in seen:
            seen.add(key)
            cleaned.append(skill)
        if len(cleaned) >= 30:
            break
    return cleaned


def _compact_jd_analysis(jd_analysis: Any) -> dict[str, Any]:
    if not isinstance(jd_analysis, dict):
        return {}

    skills: list[dict[str, Any]] = []
    for item in jd_analysis.get("skills") or []:
        if not isinstance(item, dict):
            continue
        skills.append(
            {
                "skill": _truncate_text(item.get("skill"), 80),
                "category": _truncate_text(item.get("category"), 80),
                "priority_score": item.get("priority_score"),
                "depth_required": _truncate_text(item.get("depth_required"), 120),
                "reasoning": _truncate_text(item.get("reasoning"), 180),
            }
        )
        if len(skills) >= 18:
            break

    return {
        "summary": _truncate_text(
            jd_analysis.get("summary")
            or jd_analysis.get("role_summary")
            or jd_analysis.get("overview"),
            700,
        ),
        "seniority": _truncate_text(jd_analysis.get("seniority"), 80),
        "skills": skills,
        "must_have": _compact_text_list(
            jd_analysis.get("must_have") or jd_analysis.get("required_skills"),
            max_items=12,
        ),
    }


def _compact_plan_sections(interview_plan: Any) -> list[dict[str, Any]]:
    if not isinstance(interview_plan, dict):
        return []

    sections: list[dict[str, Any]] = []
    for item in interview_plan.get("sections") or []:
        if not isinstance(item, dict):
            continue
        sections.append(
            {
                "section_name": _truncate_text(
                    item.get("section_name") or item.get("name"),
                    80,
                ),
                "skill": _truncate_text(item.get("skill"), 80),
                "allocated_mins": item.get("allocated_mins")
                or item.get("allocated_time_mins")
                or item.get("duration_mins"),
                "priority_score": item.get("priority_score"),
                "depth_required": _truncate_text(item.get("depth_required"), 120),
                "expected_signals": _compact_text_list(
                    item.get("expected_signals"),
                    max_items=6,
                    max_chars=120,
                ),
            }
        )
        if len(sections) >= 20:
            break
    return sections


def _compact_context(context: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_name": _truncate_text(context.get("candidate_name"), 120),
        "assessment_title": _truncate_text(context.get("assessment_title"), 160),
        "role_name": _truncate_text(context.get("role_name"), 120),
        "company_name": _truncate_text(context.get("company_name"), 120),
        "interview_duration_mins": context.get("interview_duration_mins"),
        "candidate_assessment_status": _truncate_text(
            context.get("candidate_assessment_status"),
            80,
        ),
        "jd_analysis": _compact_jd_analysis(context.get("jd_analysis")),
        "focus_areas": _compact_json_value(context.get("focus_areas"), 1200),
        "interview_plan_sections": _compact_plan_sections(
            context.get("interview_plan"),
        ),
        "resume_skills": _resume_skill_names(context.get("resume_parsed")),
    }


def _compact_answer_records(
    records: list[dict[str, Any]],
    *,
    max_answer_chars: int,
    max_question_chars: int = 260,
) -> list[dict[str, Any]]:
    compact_records: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        evaluation = (
            record.get("evaluation")
            if isinstance(record.get("evaluation"), dict)
            else {}
        )
        compact_records.append(
            {
                "sequence": index,
                "turn_number": record.get("turn_number"),
                "question_id": record.get("question_id"),
                "section": _truncate_text(record.get("section"), 80),
                "skill": _truncate_text(record.get("skill"), 80),
                "difficulty": _truncate_text(record.get("difficulty"), 40),
                "response_type": _truncate_text(record.get("response_type"), 40),
                "question": _truncate_text(record.get("question"), max_question_chars),
                "answer": _truncate_text(record.get("answer"), max_answer_chars),
                "live_strength": evaluation.get("strength"),
                "live_score_10": record.get("raw_score"),
                "signals_demonstrated": _compact_text_list(
                    record.get("signals_demonstrated"),
                    max_items=5,
                    max_chars=120,
                ),
                "signals_missing": _compact_text_list(
                    record.get("signals_missing"),
                    max_items=5,
                    max_chars=120,
                ),
            }
        )
    return compact_records


def _compact_violations(violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for violation in violations[:30]:
        if not isinstance(violation, dict):
            continue
        compact.append(
            {
                "turn_number": violation.get("turn_number"),
                "violation_type": violation.get("violation_type"),
                "severity": violation.get("severity"),
                "candidate_transcript": _truncate_text(
                    violation.get("candidate_transcript"),
                    300,
                ),
                "metadata": _compact_json_value(violation.get("metadata"), 500),
            }
        )
    return compact


def _compact_skill_metrics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    compact: dict[str, Any] = {}
    for skill, item in value.items():
        if not isinstance(item, dict):
            continue
        compact[str(skill)[:80]] = {
            "priority_score": item.get("priority_score"),
            "raw_score": item.get("raw_score"),
            "weighted_score": item.get("weighted_score"),
            "weight_share": item.get("weight_share"),
            "weighted_contribution": item.get("weighted_contribution"),
            "difficulty_reached": item.get("difficulty_reached"),
            "questions_asked": item.get("questions_asked"),
            "assessed": item.get("assessed"),
            "signals_demonstrated": _compact_text_list(
                item.get("signals_demonstrated"),
                max_items=5,
            ),
            "signals_missing": _compact_text_list(
                item.get("signals_missing"),
                max_items=5,
            ),
            "summary": _truncate_text(item.get("summary"), 260),
        }
        if len(compact) >= 18:
            break
    return compact


def _compact_audit_context(
    audit_context: dict[str, Any], *, include_skill_metrics: bool
) -> dict[str, Any]:
    compact = {
        "recommended_final_weighting": audit_context.get("recommended_final_weighting"),
        "transcript_answer_count": audit_context.get("transcript_answer_count"),
        "violation_count": audit_context.get("violation_count"),
        "live_signal_technical_score": audit_context.get("live_signal_technical_score"),
        "live_signal_behavioural_cultural_score": audit_context.get(
            "live_signal_behavioural_cultural_score"
        ),
        "violation_penalty_hint": audit_context.get("violation_penalty_hint"),
        "best_answer_hint": _compact_json_value(
            audit_context.get("best_answer_hint"), 700
        ),
        "weakest_answer_hint": _compact_json_value(
            audit_context.get("weakest_answer_hint"),
            700,
        ),
        "section_summary_hint": _compact_json_value(
            audit_context.get("live_signal_section_summaries"),
            1200,
        ),
        "violation_summary_hint": _compact_json_value(
            audit_context.get("violation_summary_hint"),
            900,
        ),
    }
    if include_skill_metrics:
        compact["live_signal_skill_metrics"] = _compact_skill_metrics(
            audit_context.get("live_signal_skill_metrics")
        )
    return compact


def _build_evidence_pack_blob(
    *,
    session_status: str,
    context: dict[str, Any],
    records: list[dict[str, Any]],
    violations: list[dict[str, Any]],
    audit_context: dict[str, Any],
) -> str:
    variants = (
        (900, 260, True),
        (650, 240, True),
        (420, 220, True),
        (280, 180, True),
        (180, 150, False),
        (120, 120, False),
    )
    for max_answer_chars, max_question_chars, include_skill_metrics in variants:
        pack = {
            "session_status": session_status,
            "context": _compact_context(context),
            "answer_records": _compact_answer_records(
                records,
                max_answer_chars=max_answer_chars,
                max_question_chars=max_question_chars,
            ),
            "violations": _compact_violations(violations),
            "audit_hints": _compact_audit_context(
                audit_context,
                include_skill_metrics=include_skill_metrics,
            ),
            "compaction_note": (
                "Every candidate answer is represented. Long answers and metadata "
                "may be truncated, so cite available turn numbers and concise quotes."
            ),
        }
        blob = _json_blob(pack)
        if len(blob) <= EVIDENCE_PACK_MAX_CHARS:
            return blob

    minimal_pack = {
        "session_status": session_status,
        "context": _compact_context(context),
        "answer_records": _compact_answer_records(
            records,
            max_answer_chars=80,
            max_question_chars=100,
        ),
        "violations": _compact_violations(violations),
        "audit_hints": {
            "transcript_answer_count": audit_context.get("transcript_answer_count"),
            "violation_count": audit_context.get("violation_count"),
            "live_signal_technical_score": audit_context.get(
                "live_signal_technical_score"
            ),
            "violation_penalty_hint": audit_context.get("violation_penalty_hint"),
        },
        "compaction_note": (
            "Evidence was aggressively compacted to fit the evaluator limit. "
            "Every candidate answer still has a turn number, section, skill, "
            "question, and truncated answer text."
        ),
    }
    return _json_blob(minimal_pack)


async def _generate_valid_payload(
    messages: list[dict[str, str]],
    model: type[ModelT],
) -> dict[str, Any]:
    """Call the evaluator and retry once with concrete validation feedback."""

    response_json = await evaluate(messages)
    try:
        return _parse_model_payload(response_json, model)
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        repair_messages = [
            *messages,
            {
                "role": "assistant",
                "content": response_json[:REPAIR_RESPONSE_MAX_CHARS],
            },
            {
                "role": "user",
                "content": (
                    "The previous JSON failed schema validation. Return one corrected "
                    "complete JSON object only, matching the schema exactly with no "
                    "extra keys. Preserve the evidence-backed conclusions. Validation "
                    f"error: {str(exc)[:1800]}"
                ),
            },
        ]
        repaired_json = await evaluate(repair_messages)
        return _parse_model_payload(repaired_json, model)


def _section_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _record_section_key(record: dict[str, Any]) -> str:
    return _section_key(record.get("section"))


def _is_self_intro_record(record: dict[str, Any]) -> bool:
    return _record_section_key(record) in SELF_INTRO_SECTION_KEYS


def _is_behavioural_cultural_record(record: dict[str, Any]) -> bool:
    return _record_section_key(record) in BEHAVIOURAL_CULTURAL_SECTION_KEYS


def _display_name(value: Any) -> str:
    text = str(value or "").replace("_", " ").strip()
    return text.title() if text else "General"


def _stage_evidence_blob(
    *,
    context: dict[str, Any],
    records: list[dict[str, Any]],
    violations: list[dict[str, Any]],
    audit_context: dict[str, Any],
    stage: str,
    target: dict[str, Any] | None = None,
) -> str:
    variants = ((1000, 280), (720, 240), (520, 200), (340, 160))
    for answer_chars, question_chars in variants:
        pack: dict[str, Any] = {
            "stage": stage,
            "context": _compact_context(context),
            "answer_records": _compact_answer_records(
                records,
                max_answer_chars=answer_chars,
                max_question_chars=question_chars,
            ),
            "violations": _compact_violations(violations),
            "audit_hints": _compact_audit_context(
                audit_context,
                include_skill_metrics=stage == "technical_skill",
            ),
        }
        if target:
            pack["target"] = target
        blob = _json_blob(pack)
        if len(blob) <= STAGE_EVIDENCE_MAX_CHARS:
            return blob

    return _json_blob(
        {
            "stage": stage,
            "context": _compact_context(context),
            "answer_records": _compact_answer_records(
                records,
                max_answer_chars=180,
                max_question_chars=120,
            ),
            "violations": _compact_violations(violations),
            "target": target or {},
            "compaction_note": "The evidence was compacted; do not infer missing detail.",
        }
    )


def _model_messages(
    *,
    prompt: str,
    model: type[BaseModel],
    evidence_pack: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": (
                "Return JSON matching this exact Pydantic JSON schema:\n"
                f"{_json_blob(model.model_json_schema())}\n\n"
                "Evidence pack:\n"
                f"{evidence_pack}"
            ),
        },
    ]


def _planned_technical_targets(
    *,
    records: list[dict[str, Any]],
    audit_context: dict[str, Any],
) -> list[dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    metrics = audit_context.get("live_signal_skill_metrics") or {}
    if isinstance(metrics, dict):
        for skill, metric in metrics.items():
            if not isinstance(metric, dict) or not _is_technical_section(skill, skill):
                continue
            display = str(skill).strip() or "Technical Skill"
            targets[_section_key(display)] = {
                "skill": display,
                "priority_score": metric.get("priority_score"),
                "depth_required": metric.get("depth_required"),
                "expected_signals": metric.get("signals_missing") or [],
            }

    for record in records:
        if not _is_technical_section(record.get("section"), record.get("skill")):
            continue
        display = str(record.get("skill") or record.get("section") or "Technical Skill")
        key = _section_key(display)
        targets.setdefault(
            key,
            {
                "skill": display,
                "priority_score": None,
                "depth_required": None,
                "expected_signals": [],
            },
        )

    return sorted(
        targets.values(),
        key=lambda item: float(item.get("priority_score") or 0),
        reverse=True,
    )


def _unassessed_skill_report(target: dict[str, Any]) -> dict[str, Any]:
    skill = str(target.get("skill") or "Technical Skill")
    expected = _compact_text_list(target.get("expected_signals"), max_items=12)
    report = SkillScoreReport(
        priority_score=target.get("priority_score"),
        depth_required=target.get("depth_required"),
        raw_score=None,
        weighted_score=0.0,
        weight_share=None,
        weighted_contribution=None,
        difficulty_reached=None,
        questions_asked=0,
        assessed=False,
        transcript_evidence=[],
        signals_demonstrated=[],
        signals_missing=expected or ["No direct answer was recorded for this skill."],
        summary=(
            f"{skill} was planned or required but was not directly assessed in the "
            "completed interview transcript. No competence is inferred from resume "
            "or job-description context."
        ),
    )
    return report.model_dump(mode="json")


def _unassessed_intro() -> dict[str, Any]:
    return IntroSectionEvaluation(
        section_summary=SectionSummaryReport(
            summary="No candidate self-introduction response was recorded.",
            avg_score=None,
            difficulty_reached=None,
            questions_asked=0,
            evidence=[],
            signals_demonstrated=[],
            signals_missing=["Self-introduction was not meaningfully assessed."],
            score_basis="No candidate response was available for this section.",
        ),
        strengths=[],
        concerns=["Self-introduction evidence was not available."],
    ).model_dump(mode="json")


def _unassessed_behavioural_cultural() -> dict[str, Any]:
    note = "No behavioural or cultural-fit candidate response was recorded."
    return BehaviouralCulturalEvaluation(
        behavioural_score=0.0,
        behavioural_evidence=[note],
        behavioural_summary=(
            "Behavioural capability was not meaningfully assessed; the zero score "
            "represents absent evidence, not a finding about the candidate."
        ),
        cultural_fit_score=0.0,
        cultural_fit_evidence=[note],
        cultural_fit_summary=(
            "Cultural fit was not meaningfully assessed; the zero score represents "
            "absent evidence, not a finding about the candidate."
        ),
        section_summary=SectionSummaryReport(
            summary=note,
            avg_score=None,
            difficulty_reached=None,
            questions_asked=0,
            evidence=[],
            signals_demonstrated=[],
            signals_missing=["No behavioural/cultural response was recorded."],
            score_basis="No candidate response was available for this section.",
        ),
    ).model_dump(mode="json")


async def _evaluate_intro(
    *,
    context: dict[str, Any],
    records: list[dict[str, Any]],
    violations: list[dict[str, Any]],
    audit_context: dict[str, Any],
) -> dict[str, Any]:
    if not records:
        return _unassessed_intro()
    pack = _stage_evidence_blob(
        context=context,
        records=records,
        violations=violations,
        audit_context=audit_context,
        stage="self_intro",
    )
    return await _generate_valid_payload(
        _model_messages(
            prompt=INTRO_SECTION_SYSTEM_PROMPT,
            model=IntroSectionEvaluation,
            evidence_pack=pack,
        ),
        IntroSectionEvaluation,
    )


async def _evaluate_technical_skills(
    *,
    context: dict[str, Any],
    records: list[dict[str, Any]],
    violations: list[dict[str, Any]],
    audit_context: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    targets = _planned_technical_targets(records=records, audit_context=audit_context)

    for target in targets:
        skill = str(target["skill"])
        skill_key = _section_key(skill)
        skill_records = [
            record
            for record in records
            if _is_technical_section(record.get("section"), record.get("skill"))
            and _section_key(record.get("skill") or record.get("section")) == skill_key
        ]
        if not skill_records:
            results[skill] = _unassessed_skill_report(target)
            continue

        pack = _stage_evidence_blob(
            context=context,
            records=skill_records,
            violations=violations,
            audit_context=audit_context,
            stage="technical_skill",
            target=target,
        )
        results[skill] = await _generate_valid_payload(
            _model_messages(
                prompt=TECHNICAL_SKILL_SYSTEM_PROMPT,
                model=SkillScoreReport,
                evidence_pack=pack,
            ),
            SkillScoreReport,
        )

    return results


async def _evaluate_behavioural_cultural(
    *,
    context: dict[str, Any],
    records: list[dict[str, Any]],
    violations: list[dict[str, Any]],
    audit_context: dict[str, Any],
) -> dict[str, Any]:
    if not records:
        return _unassessed_behavioural_cultural()
    pack = _stage_evidence_blob(
        context=context,
        records=records,
        violations=violations,
        audit_context=audit_context,
        stage="behavioural_cultural",
    )
    return await _generate_valid_payload(
        _model_messages(
            prompt=BEHAVIOURAL_CULTURAL_SYSTEM_PROMPT,
            model=BehaviouralCulturalEvaluation,
            evidence_pack=pack,
        ),
        BehaviouralCulturalEvaluation,
    )


def _technical_section_summaries(
    skill_scores: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    summaries: dict[str, dict[str, Any]] = {}
    for skill, score in skill_scores.items():
        summaries[skill] = SectionSummaryReport(
            summary=str(score.get("summary") or f"{skill} was not assessed."),
            avg_score=score.get("raw_score"),
            difficulty_reached=score.get("difficulty_reached"),
            questions_asked=int(score.get("questions_asked") or 0),
            evidence=list(score.get("transcript_evidence") or []),
            signals_demonstrated=list(score.get("signals_demonstrated") or []),
            signals_missing=list(score.get("signals_missing") or []),
            score_basis=(
                "Individual technical-skill evaluation based only on the questions "
                "and candidate answers assigned to this skill."
            ),
        ).model_dump(mode="json")
    return summaries


def _planned_section_summaries(
    *,
    context: dict[str, Any],
    existing: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    plan = context.get("interview_plan")
    if not isinstance(plan, dict):
        return existing
    known = {_section_key(name) for name in existing}
    for item in plan.get("sections") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("section_name") or item.get("skill") or "").strip()
        key = _section_key(name)
        if not key or key in known:
            continue
        display = _display_name(name)
        existing[display] = SectionSummaryReport(
            summary=f"{display} was planned but was not directly assessed.",
            avg_score=None,
            difficulty_reached=None,
            questions_asked=0,
            evidence=[],
            signals_demonstrated=[],
            signals_missing=[
                "No candidate answer was recorded for this planned section."
            ],
            score_basis="The interview transcript contains no assessment evidence.",
        ).model_dump(mode="json")
        known.add(key)
    return existing


def _compact_skill_for_synthesis(
    score: dict[str, Any],
    *,
    evidence_limit: int,
) -> dict[str, Any]:
    return {
        "priority_score": score.get("priority_score"),
        "depth_required": score.get("depth_required"),
        "raw_score": score.get("raw_score"),
        "weighted_score": score.get("weighted_score"),
        "questions_asked": score.get("questions_asked"),
        "assessed": score.get("assessed"),
        "difficulty_reached": score.get("difficulty_reached"),
        "summary": _truncate_text(score.get("summary"), 950),
        "signals_demonstrated": _compact_text_list(
            score.get("signals_demonstrated"), max_items=8, max_chars=160
        ),
        "signals_missing": _compact_text_list(
            score.get("signals_missing"), max_items=8, max_chars=160
        ),
        "evidence": list(score.get("transcript_evidence") or [])[:evidence_limit],
    }


def _final_synthesis_blob(
    *,
    context: dict[str, Any],
    skill_scores: dict[str, dict[str, Any]],
    intro: dict[str, Any],
    behavioural_cultural: dict[str, Any],
    violations: list[dict[str, Any]],
    audit_context: dict[str, Any],
) -> str:
    for evidence_limit in (5, 3, 2, 1):
        pack = {
            "context": _compact_context(context),
            "technical_skill_evaluations": {
                skill: _compact_skill_for_synthesis(
                    score, evidence_limit=evidence_limit
                )
                for skill, score in skill_scores.items()
            },
            "self_intro_evaluation": intro,
            "behavioural_cultural_evaluation": behavioural_cultural,
            "violations": _compact_violations(violations),
            "audit_hints": _compact_audit_context(
                audit_context,
                include_skill_metrics=False,
            ),
        }
        blob = _json_blob(pack)
        if len(blob) <= FINAL_SYNTHESIS_MAX_CHARS:
            return blob
    return _json_blob(
        {
            "context": _compact_context(context),
            "technical_skill_evaluations": {
                skill: _compact_skill_for_synthesis(score, evidence_limit=1)
                for skill, score in skill_scores.items()
            },
            "self_intro_evaluation": _compact_json_value(intro, 1800),
            "behavioural_cultural_evaluation": _compact_json_value(
                behavioural_cultural,
                2600,
            ),
            "violations": _compact_violations(violations),
        }
    )


async def _evaluate_final_synthesis(
    *,
    context: dict[str, Any],
    skill_scores: dict[str, dict[str, Any]],
    intro: dict[str, Any],
    behavioural_cultural: dict[str, Any],
    violations: list[dict[str, Any]],
    audit_context: dict[str, Any],
) -> dict[str, Any]:
    pack = _final_synthesis_blob(
        context=context,
        skill_scores=skill_scores,
        intro=intro,
        behavioural_cultural=behavioural_cultural,
        violations=violations,
        audit_context=audit_context,
    )
    return await _generate_valid_payload(
        _model_messages(
            prompt=FINAL_SYNTHESIS_SYSTEM_PROMPT,
            model=FinalSynthesisReport,
            evidence_pack=pack,
        ),
        FinalSynthesisReport,
    )


def _assemble_report(
    *,
    skill_scores: dict[str, dict[str, Any]],
    intro: dict[str, Any],
    behavioural_cultural: dict[str, Any],
    synthesis: dict[str, Any],
    context: dict[str, Any],
    audit_context: dict[str, Any],
) -> dict[str, Any]:
    section_summaries = _technical_section_summaries(skill_scores)
    section_summaries["Self Intro"] = intro["section_summary"]
    section_summaries["Behavioural & Cultural"] = behavioural_cultural[
        "section_summary"
    ]
    section_summaries = _planned_section_summaries(
        context=context,
        existing=section_summaries,
    )

    violation_summary = synthesis.get("violation_summary")
    if violation_summary is None:
        violation_summary = audit_context.get("violation_summary_hint")

    report = {
        "skill_scores": skill_scores,
        "technical_dimension_score": synthesis["technical_dimension_score"],
        "score_evidence": synthesis["score_evidence"],
        "score_summary": synthesis["score_summary"],
        "behavioural_score": behavioural_cultural["behavioural_score"],
        "behavioural_evidence": behavioural_cultural["behavioural_evidence"],
        "behavioural_summary": behavioural_cultural["behavioural_summary"],
        "cultural_fit_score": behavioural_cultural["cultural_fit_score"],
        "cultural_fit_evidence": behavioural_cultural["cultural_fit_evidence"],
        "cultural_fit_summary": behavioural_cultural["cultural_fit_summary"],
        "tone_classification_score": None,
        "tone_distribution": None,
        "section_summaries": section_summaries,
        "overall_score": synthesis["overall_score"],
        "hiring_recommendation": synthesis["hiring_recommendation"],
        "overall_narrative": synthesis["overall_narrative"],
        "strengths": synthesis["strengths"],
        "concerns": synthesis["concerns"],
        "violation_summary": violation_summary,
        "best_answer": synthesis.get("best_answer"),
        "weakest_answer": synthesis.get("weakest_answer"),
        "recommendation_reasoning": synthesis["recommendation_reasoning"],
    }
    return HolisticEvaluationReport.model_validate(report).model_dump(mode="json")


async def run_holistic_evaluation(
    candidate_assessment_id: str,
    state: dict[str, Any] | None = None,
) -> bool:
    """Run focused section evaluations and persist their final synthesis."""
    try:
        ca_uuid = uuid.UUID(candidate_assessment_id)
    except ValueError:
        logger.error("Invalid UUID for evaluation: %s", candidate_assessment_id)
        return False

    state = state or {}
    data = await get_interview_data_for_evaluation(ca_uuid)
    session = (data or {}).get("session") or {}
    context = (data or {}).get("context") or {}
    session_id_value = (
        session.get("id")
        or state.get("interview_session_id")
        or state.get("session_id")
    )
    if not session_id_value:
        logger.warning("No interview session found for %s", candidate_assessment_id)
        return False

    session_id = uuid.UUID(str(session_id_value))
    transcript = list(session.get("transcript") or state.get("transcript") or [])
    violations = list(session.get("violations") or state.get("violations") or [])
    session_status = str(session.get("status") or state.get("session_status") or "")
    if not any(
        _is_candidate_turn(turn) for turn in transcript if isinstance(turn, dict)
    ):
        logger.warning(
            "No candidate transcript turns found for %s", candidate_assessment_id
        )
        return False

    records = _candidate_answer_records(
        transcript,
        list(state.get("live_evaluations") or []),
    )
    audit_context = _build_model_audit_context(
        records=records,
        violations=violations,
        context=context,
        session_status=session_status,
    )
    intro_records = [record for record in records if _is_self_intro_record(record)]
    behavioural_records = [
        record for record in records if _is_behavioural_cultural_record(record)
    ]

    try:
        intro = await _evaluate_intro(
            context=context,
            records=intro_records,
            violations=violations,
            audit_context=audit_context,
        )
        skill_scores = await _evaluate_technical_skills(
            context=context,
            records=records,
            violations=violations,
            audit_context=audit_context,
        )
        behavioural_cultural = await _evaluate_behavioural_cultural(
            context=context,
            records=behavioural_records,
            violations=violations,
            audit_context=audit_context,
        )
        synthesis = await _evaluate_final_synthesis(
            context=context,
            skill_scores=skill_scores,
            intro=intro,
            behavioural_cultural=behavioural_cultural,
            violations=violations,
            audit_context=audit_context,
        )
        eval_data = _assemble_report(
            skill_scores=skill_scores,
            intro=intro,
            behavioural_cultural=behavioural_cultural,
            synthesis=synthesis,
            context=context,
            audit_context=audit_context,
        )
    except Exception:
        logger.exception(
            "Failed to generate staged holistic evaluation for %s",
            ca_uuid,
        )
        return False

    try:
        await save_holistic_evaluation(ca_uuid, session_id, eval_data)
    except Exception:
        logger.exception("Failed to persist holistic evaluation for %s", ca_uuid)
        return False

    logger.info("Staged holistic evaluation completed and saved for %s", ca_uuid)
    return True
