"""
InterviewOrchestrator — Bridges WebSocket ↔ LangGraph interview graph.

Manages the full lifecycle of a single interview session:
  - start_session(): runs init → opening → await_response
  - resume_session(): reconnects to an existing checkpoint
  - submit_response(): resumes graph from interrupt with candidate transcript
  - pause_session(): marks the session as paused (disconnect / tab switch)
  - unpause_session(): resumes timing after reconnection
  - get_current_state(): reads latest state from the checkpointer

The thread_id for checkpointing is the candidate_assessment_id, so any
WebSocket connection with the same assessment_id reconnects to the same
graph state — this is the core of fault tolerance.

Grace period: When a candidate disconnects, the session enters a
configurable grace period (default 5 minutes). If they reconnect within
that window, the session resumes exactly where it left off. If the
grace period expires, the session is auto-submitted.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote_plus

from langgraph.types import Command

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Grace period before auto-submitting a disconnected session
GRACE_PERIOD_SECS = 300  # 5 minutes


def _get_psycopg_uri() -> str:
    """
    Build a psycopg-compatible connection string.

    LangGraph's AsyncPostgresSaver requires psycopg (v3) format,
    not asyncpg format.
    """
    return (
        f"postgresql://{quote_plus(settings.POSTGRES_USER)}"
        f":{quote_plus(settings.POSTGRES_PASSWORD)}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}"
        f"/{settings.POSTGRES_DB}"
    )


class InterviewOrchestrator:
    """
    Orchestrates a single interview session's LangGraph lifecycle.

    One instance per WebSocket connection. The compiled graph and
    checkpointer are shared across the application lifecycle.

    Fault tolerance: The graph state is fully checkpointed to Postgres
    after each node execution. On reconnection, the orchestrator loads
    the existing checkpoint and resumes from the last interrupt point.
    """

    def __init__(
        self,
        candidate_id: str,
        assessment_id: str,
        compiled_graph: Any = None,
    ) -> None:
        self._candidate_id = candidate_id
        self._assessment_id = assessment_id
        self._graph = compiled_graph
        self._config: dict[str, Any] = {
            "configurable": {
                "thread_id": assessment_id,
            }
        }

    # ── Session lifecycle ─────────────────────────────────────────────────────

    async def start_session(self) -> dict:
        """
        Initialise and run the graph through init → opening → await_response.

        Returns the graph state after the opening monologue
        (graph is now interrupted at await_response).
        """
        logger.info(
            "Starting interview session: candidate=%s assessment=%s",
            self._candidate_id,
            self._assessment_id,
        )

        if self._graph is None:
            raise RuntimeError(
                "compiled_graph must be provided to InterviewOrchestrator"
            )

        # Provide initial state with the assessment ID
        initial_state = {
            "candidate_assessment_id": self._assessment_id,
        }

        # Run the graph — it will execute:
        #   init_session → deliver_opening → await_response (interrupt)
        result = await self._graph.ainvoke(
            initial_state,
            config=self._config,
        )

        logger.info(
            "Session started, graph interrupted at await_response: assessment=%s",
            self._assessment_id,
        )

        return result

    async def resume_session(self) -> dict | None:
        """
        Attempt to resume an existing interview session from its checkpoint.

        Returns the restored graph state if a valid checkpoint exists
        and the session is still resumable (in_progress or paused, and
        within the grace period if paused). Returns None if no valid
        checkpoint exists or the session cannot be resumed.

        On successful resume:
        - Unpauses the session (adjusts timing)
        - Returns state with the current question so the WS handler
          can re-send it to the candidate
        """
        if self._graph is None:
            return None

        try:
            snapshot = await self._graph.aget_state(self._config)
        except Exception:
            logger.exception("Failed to load checkpoint for resume")
            return None

        if not snapshot or not snapshot.values:
            logger.info(
                "No checkpoint found for assessment=%s — fresh start needed",
                self._assessment_id,
            )
            return None

        state = dict(snapshot.values)
        status = state.get("session_status", "")

        # Check if the session is in a resumable state
        if status in ("completed", "terminated", "deactivated"):
            logger.info(
                "Session %s is already %s — cannot resume",
                self._assessment_id,
                status,
            )
            return None

        # Check grace period if the session was paused (disconnected)
        grace_expires = state.get("grace_period_expires_at")
        if grace_expires:
            try:
                expires_dt = datetime.fromisoformat(grace_expires)
                now = datetime.now(UTC)
                if now > expires_dt:
                    logger.warning(
                        "Grace period expired for assessment=%s "
                        "(expired at %s, now is %s) — deactivating",
                        self._assessment_id,
                        grace_expires,
                        now.isoformat(),
                    )
                    # Mark as deactivated via graph state update
                    await self._update_state_fields(
                        {
                            "session_status": "deactivated",
                            "auto_submit_triggered": True,
                        }
                    )
                    return None
            except (ValueError, TypeError):
                logger.warning(
                    "Could not parse grace_period_expires_at: %s", grace_expires
                )

        # Check if the graph has a pending interrupt (await_response)
        has_pending_interrupt = bool(snapshot.next)

        if not has_pending_interrupt:
            logger.warning(
                "Checkpoint exists but no pending interrupt for assessment=%s "
                "— graph may be in an unexpected state",
                self._assessment_id,
            )
            return None

        # Unpause if it was paused
        if state.get("paused_at"):
            await self._unpause_timing(state)

        logger.info(
            "Session resumed from checkpoint: assessment=%s turn=%d "
            "section=%d status=%s",
            self._assessment_id,
            state.get("turn_number", 0),
            state.get("current_section_index", 0),
            status,
        )

        return state

    async def submit_response(self, transcript: str) -> dict:
        """
        Resume the graph from the await_response interrupt with the
        candidate's transcript.

        The graph will execute:
          classify_response → [handler] → (evaluate → generate_question →)
          await_response (interrupt again)

        Returns the graph state after the next interrupt.
        """
        if self._graph is None:
            raise RuntimeError("Session not started — call start_session() first")

        logger.info(
            "Submitting response: assessment=%s len=%d",
            self._assessment_id,
            len(transcript),
        )

        # Resume from interrupt with the transcript
        result = await self._graph.ainvoke(
            Command(resume=transcript),
            config=self._config,
        )

        logger.info(
            "Response processed, graph state updated: assessment=%s",
            self._assessment_id,
        )

        return result

    # ── Pause / Unpause (disconnect / reconnect timing) ───────────────────────

    async def pause_session(self) -> None:
        """
        Mark the session as paused when the candidate disconnects.

        Records the pause timestamp and sets a grace period expiration.
        The section timer is effectively frozen until unpause.
        """
        now = datetime.now(UTC)
        grace_expires = now + timedelta(seconds=GRACE_PERIOD_SECS)

        await self._update_state_fields(
            {
                "session_status": "paused",
                "paused_at": now.isoformat(),
                "grace_period_expires_at": grace_expires.isoformat(),
            }
        )

        logger.info(
            "Session paused: assessment=%s grace_expires=%s",
            self._assessment_id,
            grace_expires.isoformat(),
        )

    async def _unpause_timing(self, state: dict) -> None:
        """
        Resume timing after reconnection.

        Calculates how long the session was paused and adds it to
        total_pause_secs. Clears the paused_at and grace period fields.
        """
        paused_at_str = state.get("paused_at")
        if not paused_at_str:
            return

        try:
            paused_at = datetime.fromisoformat(paused_at_str)
            now = datetime.now(UTC)
            pause_duration = int((now - paused_at).total_seconds())
        except (ValueError, TypeError):
            pause_duration = 0

        total_pause = state.get("total_pause_secs", 0) + pause_duration

        await self._update_state_fields(
            {
                "session_status": "in_progress",
                "paused_at": None,
                "grace_period_expires_at": None,
                "total_pause_secs": total_pause,
            }
        )

        logger.info(
            "Session unpaused: assessment=%s pause_duration=%ds total_pause=%ds",
            self._assessment_id,
            pause_duration,
            total_pause,
        )

    # ── State access ──────────────────────────────────────────────────────────

    async def get_current_state(self) -> dict | None:
        """Get the current graph state from the checkpointer."""
        if self._graph is None:
            return None

        try:
            snapshot = await self._graph.aget_state(self._config)
            return dict(snapshot.values) if snapshot.values else None
        except Exception:
            logger.exception("Failed to get graph state")
            return None

    async def has_existing_session(self) -> bool:
        """
        Check if a checkpoint already exists for this assessment_id.

        Used by the WebSocket handler to decide between start and resume.
        """
        if self._graph is None:
            return False

        try:
            snapshot = await self._graph.aget_state(self._config)
            return bool(snapshot and snapshot.values)
        except Exception:
            return False

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _update_state_fields(self, updates: dict) -> None:
        """
        Update specific fields in the graph state via the checkpointer.

        Uses graph.aupdate_state() to merge partial updates into the
        existing checkpoint without re-running any nodes.
        """
        if self._graph is None:
            return

        try:
            await self._graph.aupdate_state(
                self._config,
                updates,
                as_node="await_response",
            )
        except Exception:
            logger.exception(
                "Failed to update graph state fields: %s", list(updates.keys())
            )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def assessment_id(self) -> str:
        return self._assessment_id

    @property
    def candidate_id(self) -> str:
        return self._candidate_id
