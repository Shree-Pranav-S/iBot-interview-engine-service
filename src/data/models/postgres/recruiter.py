"""Read-only recruiter mapping for interview and evaluation context."""

import uuid

from sqlalchemy import Boolean, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.data.models.postgres.base import Base
from src.data.models.postgres.mixins import TimestampMixin


class Recruiter(TimestampMixin, Base):
    """Recruiter fields consumed by the interview engine."""

    __tablename__ = "recruiters"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    full_name: Mapped[str] = mapped_column(Text, nullable=False)
    email: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(Text, nullable=False)
    company_name: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False)
