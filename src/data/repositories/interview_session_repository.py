"""Repository methods for durable interview session state."""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from src.data.clients.postgres_client import get_session_factory

SESSION_STATUSES = {
    "INITIALIZING",
    "IN_PROGRESS",
    "DISCONNECTED",
    "COMPLETED",
    "EVALUATED",
    "EVALUATION_FAILED",
    "DEACTIVATED",
    "TERMINATED",
}


def _uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _json_array(value: Any) -> str:
    return json.dumps([value], default=str)


def append_unique_json_item(
    items: Iterable[dict[str, Any]] | None,
    item: dict[str, Any],
    id_key: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Return an idempotently appended JSON array and whether it changed."""

    current = list(items or [])
    item_id = str(item.get(id_key) or "")
    if not item_id:
        raise ValueError(f"{id_key} is required for idempotent append")

    if any(str(existing.get(id_key) or "") == item_id for existing in current):
        return current, False

    return [*current, item], True


async def _fetch_one(query: str, params: dict[str, Any]) -> dict[str, Any]:
    session_factory = await get_session_factory()
    async with session_factory() as session:
        result = await session.execute(text(query), params)
        row = result.mappings().first()
        return dict(row) if row else {}


async def _fetch_one_write(query: str, params: dict[str, Any]) -> dict[str, Any]:
    session_factory = await get_session_factory()
    async with session_factory() as session:
        result = await session.execute(text(query), params)
        row = result.mappings().first()
        await session.commit()
        return dict(row) if row else {}


async def _execute(query: str, params: dict[str, Any]) -> None:
    session_factory = await get_session_factory()
    async with session_factory() as session:
        await session.execute(text(query), params)
        await session.commit()


async def get_session_by_candidate_assessment_id(
    candidate_assessment_id: str | uuid.UUID,
) -> dict[str, Any]:
    """Return the single interview_sessions row for a candidate assessment."""

    return await _fetch_one(
        """
        SELECT id, candidate_assessment_id, status, transcript, violations,
               total_elapsed_secs, total_pause_secs, reconnect_deadline,
               disconnect_count, timeout_disconnect_count,
               last_disconnected_at, session_token_expires_at,
               active_connection_id,
               created_at, last_updated_at
        FROM interview_sessions
        WHERE candidate_assessment_id = :candidate_assessment_id
        """,
        {"candidate_assessment_id": _uuid(candidate_assessment_id)},
    )


async def get_session(session_id: str | uuid.UUID) -> dict[str, Any]:
    """Return an interview_sessions row by primary key."""

    return await _fetch_one(
        """
        SELECT id, candidate_assessment_id, status, transcript, violations,
               total_elapsed_secs, total_pause_secs, reconnect_deadline,
               disconnect_count, timeout_disconnect_count,
               last_disconnected_at, session_token_expires_at,
               active_connection_id,
               created_at, last_updated_at
        FROM interview_sessions
        WHERE id = :session_id
        """,
        {"session_id": _uuid(session_id)},
    )


async def get_or_create_session(
    candidate_assessment_id: str | uuid.UUID,
) -> dict[str, Any]:
    """Create or return the one session row for candidate_assessment_id."""

    return await _fetch_one_write(
        """
        INSERT INTO interview_sessions (
            candidate_assessment_id,
            status,
            transcript,
            violations,
            total_elapsed_secs,
            total_pause_secs
        )
        VALUES (
            :candidate_assessment_id,
            'INITIALIZING',
            '[]'::jsonb,
            '[]'::jsonb,
            0,
            0
        )
        ON CONFLICT (candidate_assessment_id) DO UPDATE
        SET last_updated_at = interview_sessions.last_updated_at
        RETURNING id, candidate_assessment_id, status, transcript, violations,
                  total_elapsed_secs, total_pause_secs, reconnect_deadline,
                  disconnect_count, timeout_disconnect_count,
                  last_disconnected_at, session_token_expires_at,
                  active_connection_id,
                  created_at, last_updated_at
        """,
        {"candidate_assessment_id": _uuid(candidate_assessment_id)},
    )


async def append_transcript_turn(
    session_id: str | uuid.UUID,
    turn: dict[str, Any],
    *,
    total_elapsed_secs: int | None = None,
) -> bool:
    """Append a transcript turn once, keyed by deterministic turn_id."""

    turn_id = str(turn.get("turn_id") or "")
    if not turn_id:
        raise ValueError("turn_id is required for transcript append")

    row = await _fetch_one_write(
        """
        WITH existing AS (
            SELECT EXISTS (
                SELECT 1
                FROM interview_sessions s,
                     jsonb_array_elements(COALESCE(s.transcript, '[]'::jsonb)) elem
                WHERE s.id = :session_id
                  AND elem->>'turn_id' = :turn_id
            ) AS already_exists
        ),
        updated AS (
            UPDATE interview_sessions
            SET transcript = CASE
                    WHEN (SELECT already_exists FROM existing)
                    THEN COALESCE(transcript, '[]'::jsonb)
                    ELSE COALESCE(transcript, '[]'::jsonb) || CAST(:turn_array AS jsonb)
                END,
                total_elapsed_secs = COALESCE(
                    :total_elapsed_secs,
                    total_elapsed_secs
                ),
                last_updated_at = NOW()
            WHERE id = :session_id
            RETURNING NOT (SELECT already_exists FROM existing) AS appended
        )
        SELECT appended FROM updated
        """,
        {
            "session_id": _uuid(session_id),
            "turn_id": turn_id,
            "turn_array": _json_array(turn),
            "total_elapsed_secs": total_elapsed_secs,
        },
    )
    return bool(row.get("appended"))


async def append_violation(
    session_id: str | uuid.UUID,
    violation: dict[str, Any],
) -> bool:
    """Append a violation once, keyed by deterministic violation_id."""

    violation_id = str(violation.get("violation_id") or "")
    if not violation_id:
        raise ValueError("violation_id is required for violation append")

    row = await _fetch_one_write(
        """
        WITH existing AS (
            SELECT EXISTS (
                SELECT 1
                FROM interview_sessions s,
                     jsonb_array_elements(COALESCE(s.violations, '[]'::jsonb)) elem
                WHERE s.id = :session_id
                  AND elem->>'violation_id' = :violation_id
            ) AS already_exists
        ),
        updated AS (
            UPDATE interview_sessions
            SET violations = CASE
                    WHEN (SELECT already_exists FROM existing)
                    THEN COALESCE(violations, '[]'::jsonb)
                    ELSE COALESCE(violations, '[]'::jsonb) || CAST(:violation_array AS jsonb)
                END,
                last_updated_at = NOW()
            WHERE id = :session_id
            RETURNING NOT (SELECT already_exists FROM existing) AS appended
        )
        SELECT appended FROM updated
        """,
        {
            "session_id": _uuid(session_id),
            "violation_id": violation_id,
            "violation_array": _json_array(violation),
        },
    )
    return bool(row.get("appended"))


async def update_elapsed_time(
    session_id: str | uuid.UUID,
    *,
    elapsed_secs: int,
    total_pause_secs: int,
) -> None:
    """Persist elapsed interview time excluding pauses."""

    await _execute(
        """
        UPDATE interview_sessions
        SET total_elapsed_secs = GREATEST(
                COALESCE(total_elapsed_secs, 0),
                :elapsed_secs
            ),
            total_pause_secs = GREATEST(
                COALESCE(total_pause_secs, 0),
                :total_pause_secs
            ),
            last_updated_at = NOW()
        WHERE id = :session_id
        """,
        {
            "session_id": _uuid(session_id),
            "elapsed_secs": max(0, int(elapsed_secs)),
            "total_pause_secs": max(0, int(total_pause_secs)),
        },
    )


async def mark_session_in_progress(session_id: str | uuid.UUID) -> None:
    """Mark a session active and clear any reconnect grace deadline."""

    await _execute(
        """
        UPDATE interview_sessions
        SET status = 'IN_PROGRESS',
            reconnect_deadline = NULL,
            last_updated_at = NOW()
        WHERE id = :session_id
        """,
        {"session_id": _uuid(session_id)},
    )


async def complete_session(
    session_id: str | uuid.UUID,
    *,
    total_elapsed_secs: int,
) -> None:
    """Mark a session completed without creating final evaluations."""

    await _execute(
        """
        UPDATE interview_sessions
        SET status = 'COMPLETED',
            total_elapsed_secs = :total_elapsed_secs,
            reconnect_deadline = NULL,
            active_connection_id = NULL,
            last_updated_at = NOW()
        WHERE id = :session_id
        """,
        {
            "session_id": _uuid(session_id),
            "total_elapsed_secs": max(0, int(total_elapsed_secs)),
        },
    )


async def deactivate_session(
    session_id: str | uuid.UUID,
    *,
    reason: str,
) -> None:
    """Deactivate a session, recording the reason in violations if provided."""

    if reason:
        session_id_text = str(session_id)
        await append_violation(
            session_id,
            {
                "violation_id": f"{session_id_text}:deactivated",
                "turn_number": 0,
                "violation_type": "deactivated",
                "candidate_transcript": "",
                "severity": "high",
                "timestamp": _utc_now().isoformat(),
                "metadata": {"reason": reason},
            },
        )

    await _execute(
        """
        UPDATE interview_sessions
        SET status = 'DEACTIVATED',
            reconnect_deadline = NULL,
            active_connection_id = NULL,
            last_updated_at = NOW()
        WHERE id = :session_id
        """,
        {"session_id": _uuid(session_id)},
    )
