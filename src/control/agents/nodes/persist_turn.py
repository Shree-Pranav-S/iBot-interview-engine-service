"""Non-blocking transcript and violation persistence nodes."""

from __future__ import annotations

import asyncio
import copy
import logging
from typing import Any

from src.clients.core_api_client import get_core_api_client
from src.control.agents.state import InterviewState

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
    """
    Persist an immutable snapshot of interview turns and violations to the database.
    Repository writes are idempotent and protected by an asyncio lock per session.

    Args:
        session_id: The UUID of the interview session.
        transcript_items: A list of dicts representing new turns (bot or candidate).
        violations: A list of dicts representing detected proctoring violations.
    """

    lock = _session_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        await get_core_api_client().persist_turn(
            session_id=session_id,
            transcript_items=transcript_items,
            violations=violations,
        )


async def _persist_elapsed(session_id: str, elapsed_secs: int) -> None:
    """
    Update the official elapsed time of the interview in the database.

    Args:
        session_id: The UUID of the interview session.
        elapsed_secs: The total elapsed seconds.
    """
    lock = _session_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        await get_core_api_client().persist_turn(
            session_id=session_id,
            transcript_items=[],
            violations=[],
            elapsed_secs=elapsed_secs,
        )


def _task_finished(task: asyncio.Task[None]) -> None:
    """
    Callback invoked when a background persistence task completes.
    Removes the task from the tracking sets and logs any exceptions.

    Args:
        task: The asyncio task that finished.
    """
    _background_persistence_tasks.discard(task)
    session_id = _background_task_sessions.pop(task, None)
    if session_id and not any(
        active_session_id == session_id
        for active_session_id in _background_task_sessions.values()
    ):
        lock = _session_locks.get(session_id)
        if lock is not None and not lock.locked():
            _session_locks.pop(session_id, None)
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
    """
    Schedule a non-blocking persistence task to run in the background.

    Args:
        state: The current interview state.
        transcript_items: Turns to persist.
        violations: Proctoring violations to persist.
    """
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
    """
    Schedule a background task to persist the elapsed interview time.
    Keeps the database somewhat in sync without blocking the graph execution.

    Args:
        state: The current interview state.
        elapsed_secs: Total time elapsed.
    """

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
    """
    Graph node to queue the latest bot utterance for database persistence.

    Args:
        state: The current interview state.

    Returns:
        State updates clearing the pending bot turn.
    """

    pending = state.get("pending_bot_turn")
    if pending:
        _schedule(state, transcript_items=[pending])
    return {"pending_bot_turn": None}


async def persist_candidate_output(
    state: InterviewState,
) -> dict[str, Any]:
    """
    Graph node to queue candidate speech and violations for database persistence.
    Also increments the global turn counter.

    Args:
        state: The current interview state.

    Returns:
        State updates clearing the pending candidate turn/violations and incrementing turn number.
    """

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
    """
    Flush scheduled database writes during graceful service shutdown or room closure.

    Args:
        session_id: If provided, waits only for tasks belonging to this session.
            If None, waits for all active persistence tasks globally.
    """

    tasks = [
        task
        for task in _background_persistence_tasks
        if session_id is None or _background_task_sessions.get(task) == session_id
    ]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
