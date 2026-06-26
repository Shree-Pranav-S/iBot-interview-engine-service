import re
from statistics import mean
from typing import Any

_RECOMMENDATIONS = {"STRONG_HIRE", "HIRE", "CONSIDER", "WEAK", "NO_HIRE"}
_NON_TECH_KEYS = {
    "self_intro",
    "intro",
    "introduction",
    "behavioural",
    "behavioral",
    "behavioural_cultural",
    "behavioral_cultural",
    "cultural",
    "culture",
    "closing",
    "general",
}
_SEVERITY_PENALTY = {
    "candidate_requested_answer": 12.0,
    "unsafe_or_prompt_injection": 12.0,
    "repeated_silence": 8.0,
    "irrelevant": 4.0,
    "silence": 3.0,
    "non_answer": 3.0,
    "deactivated": 10.0,
    "candidate_insisted_skip_behavioural_cultural": 10.0,
}
_DIFFICULTY_ORDER = {"easy": 1, "medium": 2, "hard": 3}


def _section_key(name: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")


def _skill_key(name: str | None) -> str:
    return _section_key(name) or "general"


def _display_name(name: str | None) -> str:
    text = str(name or "").replace("_", " ").strip()
    return text.title() if text else "General"


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: Any, low: float = 0.0, high: float = 100.0) -> float:
    return round(max(low, min(high, _float(value))), 2)


def _score_10(value: Any) -> float:
    score = _float(value)
    if score <= 5.0:
        score *= 2.0
    return round(max(0.0, min(10.0, score)), 2)


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_text_list(value: Any, fallback: list[str] | None = None) -> list[str]:
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return cleaned or list(fallback or [])
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return list(fallback or [])


def _is_candidate_turn(turn: dict[str, Any]) -> bool:
    return str(turn.get("speaker") or "").lower() in {"candidate", "user"}


def _is_bot_turn(turn: dict[str, Any]) -> bool:
    return str(turn.get("speaker") or "").lower() in {"bot", "interviewer"}


def _is_technical_section(section: str | None, skill: str | None) -> bool:
    if skill:
        return _section_key(skill) not in _NON_TECH_KEYS
    return _section_key(section) not in _NON_TECH_KEYS


def _normalize_focus_areas(focus_areas: Any) -> dict[str, float]:
    weights: dict[str, float] = {}
    if isinstance(focus_areas, dict):
        items = focus_areas.items()
        for key, value in items:
            raw = value.get("weight_override") if isinstance(value, dict) else value
            if raw is not None:
                weights[_skill_key(str(key))] = max(1.0, _float(raw, 0.0))
        return weights

    if isinstance(focus_areas, list):
        for item in focus_areas:
            if not isinstance(item, dict):
                continue
            skill = str(item.get("skill") or "").strip()
            raw = item.get("weight_override")
            if skill and raw is not None:
                weights[_skill_key(skill)] = max(1.0, _float(raw, 0.0))
    return weights


def _jd_skill_priorities(jd_analysis: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(jd_analysis, dict):
        return {}

    priorities: dict[str, dict[str, Any]] = {}
    for item in _as_list(jd_analysis.get("skills")):
        if not isinstance(item, dict):
            continue
        skill = str(item.get("skill") or "").strip()
        if not skill:
            continue
        priorities[_skill_key(skill)] = {
            "skill": skill,
            "priority_score": _float(item.get("priority_score"), 5.0),
            "depth_required": item.get("depth_required"),
            "reasoning": item.get("reasoning"),
        }
    return priorities


def _plan_sections(interview_plan: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(interview_plan, dict):
        return {}

    sections: dict[str, dict[str, Any]] = {}
    for item in _as_list(interview_plan.get("sections")):
        if not isinstance(item, dict):
            continue
        section_name = str(item.get("section_name") or item.get("name") or "").strip()
        skill = item.get("skill")
        key = _skill_key(str(skill or section_name))
        if not key:
            continue
        sections[key] = {
            "section_name": section_name or _display_name(skill),
            "skill": skill,
            "priority_score": item.get("priority_score"),
            "expected_signals": item.get("expected_signals") or [],
            "allocated_mins": item.get("allocated_mins"),
        }
    return sections


def _difficulty_max(values: list[Any]) -> str | None:
    best: str | None = None
    best_rank = 0
    for value in values:
        label = str(value or "").lower()
        rank = _DIFFICULTY_ORDER.get(label, 0)
        if rank > best_rank:
            best = label
            best_rank = rank
    return best


def _candidate_answer_records(
    transcript: list[dict[str, Any]],
    live_evaluations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered_evaluations = [item for item in live_evaluations if isinstance(item, dict)]
    eval_by_question = {
        str(item.get("question_id") or ""): item
        for item in ordered_evaluations
        if item.get("question_id")
    }
    eval_by_turn: dict[int, dict[str, Any]] = {}
    question_ids_with_turn_scoped_eval: set[str] = set()
    for item in ordered_evaluations:
        parts = str(item.get("evaluation_id") or "").split(":")
        if len(parts) < 2:
            continue
        try:
            turn_number = int(parts[1])
        except ValueError:
            continue
        eval_by_turn[turn_number] = item
        if item.get("question_id"):
            question_ids_with_turn_scoped_eval.add(str(item.get("question_id")))

    eval_index = 0
    last_question: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []

    for turn in transcript:
        if not isinstance(turn, dict):
            continue
        if _is_bot_turn(turn) and (
            turn.get("question_id")
            or (turn.get("metadata") or {}).get("message_type") == "question"
        ):
            last_question = turn
            continue

        if not _is_candidate_turn(turn):
            continue

        turn_number = int(turn.get("turn_number") or 0)
        question_id = str(turn.get("question_id") or "")
        evaluation = eval_by_turn.get(turn_number)
        allow_index_fallback = question_id not in question_ids_with_turn_scoped_eval
        if evaluation is None and allow_index_fallback:
            evaluation = eval_by_question.get(question_id)
        if evaluation is None and allow_index_fallback:
            evaluation = (
                ordered_evaluations[eval_index]
                if eval_index < len(ordered_evaluations)
                else {}
            )
        if evaluation is None:
            evaluation = {}
        eval_index += 1
        question = last_question or {}
        answer_text = str(turn.get("text") or "").strip()
        response_type = str((turn.get("metadata") or {}).get("response_type") or "")
        section = str(turn.get("section") or evaluation.get("section") or "general")
        skill = turn.get("skill") or evaluation.get("skill") or question.get("skill")
        difficulty = (
            turn.get("difficulty")
            or question.get("difficulty")
            or evaluation.get("recommended_difficulty")
        )
        raw_score = _score_10(evaluation.get("provisional_score"))
        if raw_score <= 0 and answer_text and response_type in {"", "answer"}:
            raw_score = 4.0
        records.append(
            {
                "turn_number": turn_number,
                "question_id": question_id or evaluation.get("question_id"),
                "question": str(
                    question.get("text")
                    or question.get("question_text")
                    or turn.get("question_text")
                    or ""
                ),
                "answer": answer_text,
                "response_type": response_type,
                "section": section,
                "skill": skill,
                "difficulty": difficulty,
                "raw_score": raw_score,
                "score_percent": round(raw_score * 10.0, 2),
                "signals_demonstrated": _as_text_list(
                    evaluation.get("signals_observed")
                    or evaluation.get("signals_demonstrated")
                ),
                "signals_missing": _as_text_list(evaluation.get("signals_missing")),
                "evaluation": evaluation,
            }
        )

    return records


def _group_records(
    records: list[dict[str, Any]],
    key_name: str,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        raw_key = record.get(key_name) or record.get("section") or "general"
        grouped.setdefault(_skill_key(str(raw_key)), []).append(record)
    return grouped


def _effective_priority(
    skill_key: str,
    jd_priorities: dict[str, dict[str, Any]],
    plan_sections: dict[str, dict[str, Any]],
    focus_weights: dict[str, float],
) -> float:
    if skill_key in focus_weights:
        return focus_weights[skill_key]
    if skill_key in plan_sections and plan_sections[skill_key].get("priority_score"):
        return _float(plan_sections[skill_key].get("priority_score"), 5.0)
    if skill_key in jd_priorities:
        return _float(jd_priorities[skill_key].get("priority_score"), 5.0)
    return 5.0


def _merge_signals(records: list[dict[str, Any]], field: str) -> list[str]:
    values: list[str] = []
    for record in records:
        values.extend(_as_text_list(record.get(field)))
    seen: set[str] = set()
    result: list[str] = []
    for item in values:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result[:10]


def _build_skill_scores(
    records: list[dict[str, Any]],
    context: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], float, list[str]]:
    jd_priorities = _jd_skill_priorities(context.get("jd_analysis"))
    plan = _plan_sections(context.get("interview_plan"))
    focus = _normalize_focus_areas(context.get("focus_areas"))
    grouped_by_skill = _group_records(
        [
            record
            for record in records
            if _is_technical_section(record.get("section"), record.get("skill"))
        ],
        "skill",
    )

    planned_technical_keys = {
        key
        for key, item in plan.items()
        if _is_technical_section(item.get("section_name"), item.get("skill"))
    }
    keys = set(jd_priorities) | planned_technical_keys | set(grouped_by_skill)
    if not keys and records:
        keys = {
            _skill_key(record.get("skill") or record.get("section"))
            for record in records
            if record.get("answer")
        }

    assessed_weights: dict[str, float] = {}
    for key in keys:
        items = grouped_by_skill.get(key, [])
        if items:
            assessed_weights[key] = _effective_priority(key, jd_priorities, plan, focus)
    total_weight = sum(assessed_weights.values())

    skill_scores: dict[str, dict[str, Any]] = {}
    for key in sorted(
        keys,
        key=lambda item: _effective_priority(item, jd_priorities, plan, focus),
        reverse=True,
    ):
        items = grouped_by_skill.get(key, [])
        priority = _effective_priority(key, jd_priorities, plan, focus)
        jd_item = jd_priorities.get(key, {})
        plan_item = plan.get(key, {})
        display = str(
            jd_item.get("skill")
            or plan_item.get("skill")
            or plan_item.get("section_name")
            or _display_name(key)
        )
        expected = _as_text_list(plan_item.get("expected_signals"))
        missing = _merge_signals(items, "signals_missing")
        if expected:
            observed_lower = {
                signal.lower()
                for signal in _merge_signals(items, "signals_demonstrated")
            }
            missing_expected = [
                signal for signal in expected if signal.lower() not in observed_lower
            ]
            missing = [*missing, *missing_expected]

        if items:
            raw_score = round(
                mean([_float(item.get("raw_score")) for item in items]), 2
            )
            score_percent = round(raw_score * 10.0, 2)
            weight_share = (
                round((assessed_weights.get(key, 0.0) / total_weight) * 100.0, 2)
                if total_weight
                else None
            )
            contribution = (
                round(score_percent * (weight_share or 0.0) / 100.0, 2)
                if weight_share is not None
                else None
            )
            signals = _merge_signals(items, "signals_demonstrated")
            difficulty = _difficulty_max([item.get("difficulty") for item in items])
            summary = (
                f"{display} was assessed across {len(items)} answer(s). "
                f"Average technical score was {raw_score}/10 at priority {priority}/10."
            )
            if jd_item.get("depth_required"):
                summary += f" JD depth required: {jd_item['depth_required']}."
        else:
            raw_score = None
            score_percent = 0.0
            weight_share = None
            contribution = None
            signals = []
            difficulty = None
            summary = (
                f"{display} is present in the JD/interview plan with priority "
                f"{priority}/10, but it was not directly assessed in the transcript."
            )

        skill_scores[display] = {
            "priority_score": round(priority, 2),
            "raw_score": raw_score,
            "weighted_score": score_percent,
            "weight_share": weight_share,
            "weighted_contribution": contribution,
            "difficulty_reached": difficulty,
            "questions_asked": len(items),
            "assessed": bool(items),
            "signals_demonstrated": signals,
            "signals_missing": missing[:10],
            "summary": summary,
        }

    if total_weight:
        technical_score = round(
            sum(
                _float(skill_scores[name].get("weighted_contribution"))
                for name in skill_scores
                if skill_scores[name].get("assessed")
            ),
            2,
        )
    else:
        assessed_records = [
            record
            for record in records
            if _is_technical_section(record.get("section"), record.get("skill"))
        ]
        technical_score = (
            round(
                mean([_float(item.get("score_percent")) for item in assessed_records]),
                2,
            )
            if assessed_records
            else 0.0
        )

    evidence = []
    for record in sorted(
        records, key=lambda item: _float(item.get("score_percent")), reverse=True
    ):
        if not _is_technical_section(record.get("section"), record.get("skill")):
            continue
        answer = str(record.get("answer") or "")
        if not answer:
            continue
        skill = record.get("skill") or record.get("section") or "technical skill"
        evidence.append(
            f"{skill}: scored {record.get('raw_score')}/10 for answer on turn {record.get('turn_number')}."
        )
        if len(evidence) >= 6:
            break

    return skill_scores, _clamp(technical_score), evidence


def _section_summaries(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped = _group_records(records, "section")
    summaries: dict[str, dict[str, Any]] = {}
    for key, items in grouped.items():
        display = _display_name(key)
        scores = [_float(item.get("raw_score")) for item in items if item.get("answer")]
        avg_score = round(mean(scores), 2) if scores else None
        difficulty = _difficulty_max([item.get("difficulty") for item in items])
        signals = _merge_signals(items, "signals_demonstrated")
        missing = _merge_signals(items, "signals_missing")
        summary_parts = [
            f"{display} covered {len(items)} candidate answer(s).",
        ]
        if avg_score is not None:
            summary_parts.append(f"Average score was {avg_score}/10.")
        if signals:
            summary_parts.append(f"Observed: {', '.join(signals[:3])}.")
        if missing:
            summary_parts.append(f"Missing: {', '.join(missing[:3])}.")
        summaries[display] = {
            "summary": " ".join(summary_parts),
            "avg_score": avg_score,
            "difficulty_reached": difficulty,
            "questions_asked": len(items),
        }
    return summaries


def _dimension_score(records: list[dict[str, Any]], sections: set[str]) -> float | None:
    selected = [
        _float(record.get("score_percent"))
        for record in records
        if _section_key(record.get("section")) in sections
    ]
    if not selected:
        return None
    return _clamp(mean(selected))


def _violation_summary(
    violations: list[dict[str, Any]],
    session_status: str | None,
) -> dict[str, Any] | None:
    if not violations and session_status != "TERMINATED":
        return None
    total_irrelevant = sum(
        1
        for item in violations
        if str(item.get("violation_type") or "")
        in {"irrelevant", "candidate_requested_answer"}
    )
    total_silences = sum(
        1 for item in violations if "silence" in str(item.get("violation_type") or "")
    )
    return {
        "total_irrelevant": total_irrelevant,
        "total_silences": total_silences,
        "terminated_early": session_status == "TERMINATED",
        "entries": violations,
    }


def _violation_penalty(violations: list[dict[str, Any]]) -> float:
    penalty = 0.0
    for item in violations:
        violation_type = str(item.get("violation_type") or "")
        penalty += _SEVERITY_PENALTY.get(violation_type, 2.0)
    return min(20.0, penalty)


def _highlight(records: list[dict[str, Any]], *, best: bool) -> dict[str, Any] | None:
    scored = [record for record in records if record.get("answer")]
    if not scored:
        return None
    selected = (max if best else min)(
        scored,
        key=lambda item: _float(item.get("raw_score")),
    )
    quality = "strongest" if best else "weakest"
    return {
        "question": str(selected.get("question") or "Question text unavailable."),
        "turn_number": int(selected.get("turn_number") or 0),
        "section": _display_name(str(selected.get("section") or "General")),
        "difficulty_at_time": selected.get("difficulty"),
        "reason": (
            f"Selected as the {quality} answer with score "
            f"{selected.get('raw_score')}/10. "
            f"Signals observed: {', '.join(_as_text_list(selected.get('signals_demonstrated'))[:3]) or 'none recorded'}. "
            f"Signals missing: {', '.join(_as_text_list(selected.get('signals_missing'))[:3]) or 'none recorded'}."
        ),
    }


def _build_transcript_text(transcript: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for index, turn in enumerate(transcript, start=1):
        if not isinstance(turn, dict):
            continue
        speaker = "Interviewer" if _is_bot_turn(turn) else "Candidate"
        turn_number = turn.get("turn_number") or index
        section = turn.get("section") or "unknown"
        skill = turn.get("skill") or "none"
        text = str(turn.get("text") or "").strip()
        if text:
            lines.append(
                f"[{speaker} Turn {turn_number} | Section: {section} | Skill: {skill}]: {text}"
            )
    return "\n".join(lines)


def _build_records_text(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for record in records:
        lines.append(
            "[Answer Turn {turn} | Section {section} | Skill {skill}]: "
            "Score {score}/10; observed={observed}; missing={missing}; answer={answer}".format(
                turn=record.get("turn_number") or 0,
                section=record.get("section") or "general",
                skill=record.get("skill") or "none",
                score=record.get("raw_score") or 0,
                observed=record.get("signals_demonstrated") or [],
                missing=record.get("signals_missing") or [],
                answer=str(record.get("answer") or "")[:800],
            )
        )
    return "\n".join(lines)


def _build_model_audit_context(
    *,
    records: list[dict[str, Any]],
    violations: list[dict[str, Any]],
    context: dict[str, Any],
    session_status: str | None,
) -> dict[str, Any]:
    skill_metrics, technical_score, technical_evidence = _build_skill_scores(
        records,
        context,
    )
    behavioural_cultural_sections = {
        "behavioural_cultural",
        "behavioral_cultural",
        "behavioural",
        "behavioral",
        "cultural",
        "culture",
    }
    return {
        "recommended_final_weighting": {
            "technical_dimension_score": "70-80 percent of overall decision weight",
            "behavioural_score": "10-15 percent of overall decision weight",
            "cultural_fit_score": "10-15 percent of overall decision weight",
            "violations": "apply recommendation caps or penalties after evidence scoring",
        },
        "transcript_answer_count": len(records),
        "violation_count": len(violations),
        "live_signal_skill_metrics": skill_metrics,
        "live_signal_technical_score": technical_score,
        "live_signal_technical_evidence": technical_evidence,
        "live_signal_section_summaries": _section_summaries(records),
        "live_signal_behavioural_cultural_score": _dimension_score(
            records,
            behavioural_cultural_sections,
        ),
        "violation_penalty_hint": _violation_penalty(violations),
        "violation_summary_hint": _violation_summary(violations, session_status),
        "best_answer_hint": _highlight(records, best=True),
        "weakest_answer_hint": _highlight(records, best=False),
    }
