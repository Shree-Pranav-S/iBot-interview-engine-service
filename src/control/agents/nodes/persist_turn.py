"""Non-blocking transcript and violation persistence nodes."""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any

from src.control.agents.state import InterviewState
from src.data.repositories import interview_session_repository

logger = logging.getLogger(__name__)
_background_persistence_tasks: set[asyncio.Task[None]] = set()
_background_task_sessions: dict[asyncio.Task[None], str] = {}
_session_locks: dict[str, asyncio.Lock] = {}


async def persist_turn(
    *,
    session_id: str,
    transcript_items: list[dict[str, Any]],
    violations: list[dict[str, Any]],
) -> None:
    """Persist an immutable snapshot; repository writes are idempotent."""

    lock = _session_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        for item in transcript_items:
            await interview_session_repository.append_transcript_turn(
                session_id,
                item,
            )
        for violation in violations:
            await interview_session_repository.append_violation(
                session_id,
                violation,
            )


async def _persist_elapsed(session_id: str, elapsed_secs: int) -> None:
    lock = _session_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        await interview_session_repository.update_elapsed_time(
            session_id,
            elapsed_secs=elapsed_secs,
            total_pause_secs=0,
        )


def _task_finished(task: asyncio.Task[None]) -> None:
    _background_persistence_tasks.discard(task)
    _background_task_sessions.pop(task, None)
    if task.cancelled():
        logger.warning("Background interview-turn persistence was cancelled")
        return
    error = task.exception()
    if error is not None:
        logger.error(
            "Background interview-turn persistence failed",
            exc_info=(type(error), error, error.__traceback__),
        )


def _schedule(
    state: InterviewState,
    *,
    transcript_items: list[dict[str, Any]],
    violations: list[dict[str, Any]] | None = None,
) -> None:
    if not transcript_items and not violations:
        return
    task = asyncio.create_task(
        persist_turn(
            session_id=str(state["interview_session_id"]),
            transcript_items=copy.deepcopy(transcript_items),
            violations=copy.deepcopy(violations or []),
        )
    )
    _background_persistence_tasks.add(task)
    _background_task_sessions[task] = str(state["interview_session_id"])
    task.add_done_callback(_task_finished)


def schedule_elapsed_persistence(
    state: InterviewState,
    *,
    elapsed_secs: int,
) -> None:
    """Persist timing off the graph response path."""

    task = asyncio.create_task(
        _persist_elapsed(
            str(state["interview_session_id"]),
            max(0, int(elapsed_secs)),
        )
    )
    _background_persistence_tasks.add(task)
    _background_task_sessions[task] = str(state["interview_session_id"])
    task.add_done_callback(_task_finished)


async def persist_bot_output(state: InterviewState) -> dict[str, Any]:
    """Queue the latest bot utterance from LangGraph's active event loop."""

    pending = state.get("pending_bot_turn")
    if pending:
        _schedule(state, transcript_items=[pending])
    return {"pending_bot_turn": None}


async def persist_candidate_output(
    state: InterviewState,
) -> dict[str, Any]:
    """Queue candidate speech and violations from the active event loop."""

    pending = state.get("pending_candidate_turn")
    violations = list(state.get("violations_to_persist") or [])
    _schedule(
        state,
        transcript_items=[pending] if pending else [],
        violations=violations,
    )
    return {
        "pending_candidate_turn": None,
        "violations_to_persist": [],
        "turn_number": int(state.get("turn_number") or 1) + 1,
    }


async def drain_background_persistence(
    session_id: str | None = None,
) -> None:
    """Flush scheduled writes during graceful service shutdown."""

    tasks = [
        task
        for task in _background_persistence_tasks
        if session_id is None or _background_task_sessions.get(task) == session_id
    ]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
