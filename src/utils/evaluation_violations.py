"""Deterministic proctoring policy for holistic interview evaluation."""

from __future__ import annotations

import copy
from collections.abc import Iterable
from typing import Any

SINGLETON_PROCTORING_SEVERITIES: dict[str, str] = {
    "tab_switch": "low",
    "face_absent": "high",
    "multiple_faces": "high",
}

_FACE_VIOLATION_TYPES = {"face_absent", "multiple_faces"}


def _metadata(violation: dict[str, Any]) -> dict[str, Any]:
    value = violation.get("metadata")
    return dict(value) if isinstance(value, dict) else {}


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _positive_int_values(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        parsed = _non_negative_int(item)
        if parsed is not None and parsed > 0:
            result.append(parsed)
    return result


def _episode_durations(metadata: dict[str, Any]) -> list[int]:
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list):
        return []
    durations: list[int] = []
    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        candidates = [
            parsed
            for key in ("observed_duration_ms", "max_observed_duration_ms")
            if (parsed := _non_negative_int(episode.get(key))) is not None
            and parsed > 0
        ]
        if candidates:
            durations.append(max(candidates))
    return durations


def is_singleton_proctoring_violation(violation: dict[str, Any]) -> bool:
    """Return whether the record belongs to a score-once policy category."""

    violation_type = str(violation.get("violation_type") or "").strip()
    return violation_type in SINGLETON_PROCTORING_SEVERITIES


def violation_affects_evaluation(violation: dict[str, Any]) -> bool:
    """Apply the authoritative inclusion policy for an evaluation record."""

    if is_singleton_proctoring_violation(violation):
        # Older face records were persisted as review-only. The current policy
        # is authoritative and intentionally upgrades those historical records.
        return True
    return _metadata(violation).get("affects_evaluation") is not False


def _recorded_event(violation: dict[str, Any]) -> dict[str, Any]:
    """Keep source evidence without changing the original stored record."""

    return {
        "violation_id": violation.get("violation_id"),
        "timestamp": violation.get("timestamp"),
        "severity": violation.get("severity"),
        "metadata": _metadata(violation),
    }


def _aggregate_singleton_category(
    violation_type: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collapse repeated records while preserving their reporting evidence."""

    aggregate = copy.deepcopy(records[0])
    aggregate["violation_type"] = violation_type
    aggregate["severity"] = SINGLETON_PROCTORING_SEVERITIES[violation_type]
    metadata = _metadata(aggregate)

    occurrence_candidates = [len(records)]
    tab_switch_candidates = [len(records)]
    observed_durations: list[int] = []
    explicit_total_durations: list[int] = []
    max_face_counts: list[int] = []
    termination_triggered = False
    termination_reason: str | None = None
    termination_duration_candidates: list[int] = []

    for record in records:
        record_metadata = _metadata(record)
        parsed_occurrence_count = _non_negative_int(
            record_metadata.get("occurrence_count")
        )
        if parsed_occurrence_count is not None:
            occurrence_candidates.append(parsed_occurrence_count)
        for key in ("tab_switch_count", "count"):
            parsed = _non_negative_int(record_metadata.get(key))
            if parsed is not None:
                tab_switch_candidates.append(parsed)

        recorded_durations = _positive_int_values(
            record_metadata.get("observed_durations_ms")
        )
        if not recorded_durations:
            recorded_durations = _episode_durations(record_metadata)
        if recorded_durations:
            observed_durations.extend(recorded_durations)
        else:
            duration_candidates = [
                parsed
                for key in ("observed_duration_ms", "max_observed_duration_ms")
                if (parsed := _non_negative_int(record_metadata.get(key))) is not None
                and parsed > 0
            ]
            if duration_candidates:
                observed_durations.append(max(duration_candidates))
        explicit_total = _non_negative_int(
            record_metadata.get("total_observed_duration_ms")
        )
        if explicit_total is not None and explicit_total > 0:
            explicit_total_durations.append(explicit_total)
        face_count = _non_negative_int(record_metadata.get("max_face_count"))
        if face_count is not None:
            max_face_counts.append(face_count)
        episodes = record_metadata.get("episodes")
        if isinstance(episodes, list):
            for episode in episodes:
                if not isinstance(episode, dict):
                    continue
                face_count = _non_negative_int(episode.get("max_face_count"))
                if face_count is not None:
                    max_face_counts.append(face_count)
        if bool(record_metadata.get("termination_triggered")):
            termination_triggered = True
        raw_termination_reason = str(
            record_metadata.get("termination_reason") or ""
        ).strip()
        if raw_termination_reason:
            termination_reason = raw_termination_reason
        termination_duration = _non_negative_int(
            record_metadata.get("termination_duration_ms")
        )
        if termination_duration is not None:
            termination_duration_candidates.append(termination_duration)

    occurrence_count = max(occurrence_candidates)
    metadata.update(
        {
            "affects_evaluation": True,
            "scoring_policy": "once_per_category",
            "scored_occurrence_count": 1,
            "occurrence_count": occurrence_count,
            "recorded_events": [_recorded_event(record) for record in records],
            "termination_triggered": termination_triggered,
            "termination_reason": termination_reason,
        }
    )
    if termination_duration_candidates:
        metadata["termination_duration_ms"] = max(termination_duration_candidates)

    if violation_type == "tab_switch":
        metadata["tab_switch_count"] = max(
            occurrence_count,
            *tab_switch_candidates,
        )
    else:
        # A single stored face record may be updated as an episode continues;
        # legacy data may contain multiple records. Both forms remain visible.
        if observed_durations:
            metadata["observed_durations_ms"] = observed_durations
            metadata["max_observed_duration_ms"] = max(observed_durations)
            metadata["observed_duration_ms"] = max(observed_durations)
            metadata["total_observed_duration_ms"] = max(
                [sum(observed_durations), *explicit_total_durations]
            )
        elif explicit_total_durations:
            metadata["total_observed_duration_ms"] = max(explicit_total_durations)
        if max_face_counts:
            metadata["max_face_count"] = max(max_face_counts)

    aggregate["metadata"] = metadata
    return aggregate


def normalize_evaluation_violations(
    violations: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filter evaluation records and score policy categories only once."""

    source = [copy.deepcopy(item) for item in violations]
    singleton_groups: dict[str, list[dict[str, Any]]] = {}
    for item in source:
        violation_type = str(item.get("violation_type") or "").strip()
        if violation_type in SINGLETON_PROCTORING_SEVERITIES:
            singleton_groups.setdefault(violation_type, []).append(item)

    normalized: list[dict[str, Any]] = []
    emitted: set[str] = set()
    for item in source:
        violation_type = str(item.get("violation_type") or "").strip()
        if violation_type in SINGLETON_PROCTORING_SEVERITIES:
            if violation_type in emitted:
                continue
            normalized.append(
                _aggregate_singleton_category(
                    violation_type,
                    singleton_groups[violation_type],
                )
            )
            emitted.add(violation_type)
            continue
        if violation_affects_evaluation(item):
            normalized.append(item)
    return normalized


def singleton_policy_minimum_counts(
    violations: Iterable[dict[str, Any]],
) -> dict[str, int]:
    """Return minimum evaluator counts required by authoritative policy."""

    categories = {str(item.get("violation_type") or "").strip() for item in violations}
    return {
        "low": int("tab_switch" in categories),
        "medium": 0,
        "high": sum(item in categories for item in _FACE_VIOLATION_TYPES),
        "critical": 0,
    }


def violation_category_details(
    violations: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build deterministic recruiter-visible details for policy categories."""

    details: list[dict[str, Any]] = []
    for violation in violations:
        violation_type = str(violation.get("violation_type") or "").strip()
        if violation_type not in SINGLETON_PROCTORING_SEVERITIES:
            continue
        metadata = _metadata(violation)
        detail: dict[str, Any] = {
            "violation_type": violation_type,
            "severity": SINGLETON_PROCTORING_SEVERITIES[violation_type],
            "scored_occurrence_count": 1,
            "occurrence_count": max(
                1,
                _non_negative_int(metadata.get("occurrence_count")) or 1,
            ),
            "timestamp": violation.get("timestamp"),
            "termination_triggered": bool(metadata.get("termination_triggered")),
            "termination_reason": metadata.get("termination_reason"),
            "metadata": metadata,
        }
        if violation_type == "tab_switch":
            detail["tab_switch_count"] = max(
                1,
                _non_negative_int(metadata.get("tab_switch_count")) or 1,
            )
        else:
            detail.update(
                {
                    "observed_duration_ms": _non_negative_int(
                        metadata.get("observed_duration_ms")
                    ),
                    "max_observed_duration_ms": _non_negative_int(
                        metadata.get("max_observed_duration_ms")
                    ),
                    "total_observed_duration_ms": _non_negative_int(
                        metadata.get("total_observed_duration_ms")
                    ),
                    "observed_durations_ms": _positive_int_values(
                        metadata.get("observed_durations_ms")
                    ),
                    "termination_duration_ms": _non_negative_int(
                        metadata.get("termination_duration_ms")
                    ),
                    "max_face_count": _non_negative_int(metadata.get("max_face_count")),
                }
            )
        details.append(detail)
    return details


def deterministic_violation_evidence(
    violations: Iterable[dict[str, Any]],
) -> list[str]:
    """Create stable evidence text for existing recruiter report renderers."""

    evidence: list[str] = []
    for detail in violation_category_details(violations):
        violation_type = str(detail["violation_type"])
        if violation_type == "tab_switch":
            count = int(detail["tab_switch_count"])
            evidence.append(
                f"Tab switching was detected {count} time{'s' if count != 1 else ''}; "
                "it was scored once as a low-severity violation."
            )
            continue

        duration_ms = (
            detail.get("max_observed_duration_ms")
            or detail.get("observed_duration_ms")
            or detail.get("total_observed_duration_ms")
        )
        duration_text = (
            f" for {float(duration_ms) / 1000:.1f} seconds"
            if isinstance(duration_ms, int) and duration_ms > 0
            else ""
        )
        if violation_type == "face_absent":
            evidence.append(
                "The candidate was not visible"
                f"{duration_text}; this category was scored once as a "
                "high-severity violation."
            )
        else:
            evidence.append(
                "Multiple faces were detected"
                f"{duration_text}; this category was scored once as a "
                "high-severity violation."
            )
    return evidence
