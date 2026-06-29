"""Repository-layer transaction boundary for interview workflows."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from types import TracebackType

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.data.clients.postgres_client import get_session_factory
from src.data.repositories.assessment_context_repository import (
    AssessmentContextRepository,
)
from src.data.repositories.candidate_session_repository import (
    CandidateSessionRepository,
)
from src.data.repositories.evaluation_repository import EvaluationRepository
from src.data.repositories.event_logs_repository import EventLogsRepository
from src.data.repositories.interview_session_repository import (
    InterviewSessionRepository,
)

SessionFactoryProvider = Callable[
    [],
    Awaitable[async_sessionmaker[AsyncSession]],
]


class InterviewUnitOfWork:
    """Own one database session and expose transaction-scoped repositories."""

    def __init__(
        self,
        session_factory_provider: SessionFactoryProvider = get_session_factory,
    ) -> None:
        self._session_factory_provider = session_factory_provider
        self._session: AsyncSession | None = None
        self.assessment_context: AssessmentContextRepository
        self.candidate_sessions: CandidateSessionRepository
        self.evaluations: EvaluationRepository
        self.event_logs: EventLogsRepository
        self.interview_sessions: InterviewSessionRepository

    async def __aenter__(self) -> InterviewUnitOfWork:
        session_factory = await self._session_factory_provider()
        self._session = session_factory()
        await self._session.begin()
        self.assessment_context = AssessmentContextRepository(self._session)
        self.candidate_sessions = CandidateSessionRepository(self._session)
        self.evaluations = EvaluationRepository(self._session)
        self.event_logs = EventLogsRepository(self._session)
        self.interview_sessions = InterviewSessionRepository(self._session)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._session is None:
            return
        try:
            if exc_type is None:
                try:
                    await self._session.commit()
                except Exception:
                    await self._session.rollback()
                    raise
            else:
                await self._session.rollback()
        finally:
            await self._session.close()
            self._session = None
