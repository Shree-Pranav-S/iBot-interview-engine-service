"""Application settings for interview-engine-service."""

from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed service configuration."""

    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "interview-engine-service"
    APP_ENV: str = Field(default="development")

    # Deepgram speech services.
    DEEPGRAM_API_KEY: str = Field(default="")
    DEEPGRAM_STT_MODEL: str = Field(default="nova-3")
    DEEPGRAM_TTS_MODEL: str = Field(default="aura-2-andromeda-en")

    # Groq models used by latency-sensitive graph nodes.
    GROQ_API_KEY: str = Field(default="")
    FALLBACK_GROQ_API_KEY: str = Field(default="")
    GROQ_EVALUATION_API_KEY: str = Field(default="")
    GROQ_QUESTION_API_KEY: str = Field(default="")
    GROQ_QUESTION_MODEL: str = Field(default="llama-3.3-70b-versatile")
    GROQ_QUESTION_MAX_TOKENS: int = Field(default=256)
    GROQ_QUESTION_TEMPERATURE: float = Field(default=0.15)
    GROQ_QUESTION_TIMEOUT_SECS: float = Field(default=15.0)
    GROQ_LIVE_EVAL_MODEL: str = Field(default="llama-3.1-8b-instant")
    GROQ_LIVE_EVAL_MAX_TOKENS: int = Field(default=256)
    GROQ_LIVE_EVAL_TEMPERATURE: float = Field(default=0.0)
    GROQ_CLASSIFY_MODEL: str = Field(default="llama-3.1-8b-instant")
    GROQ_CLASSIFY_MAX_TOKENS: int = Field(default=256)
    GROQ_CLASSIFY_TEMPERATURE: float = Field(default=0.0)

    # NVIDIA NIM one-shot holistic evaluation.
    NVIDIA_NIM_API_KEY: str = Field(default="")
    NVIDIA_NIM_BASE_URL: str = Field(
        default="https://integrate.api.nvidia.com/v1",
    )
    NVIDIA_NIM_MODEL: str = Field(
        default="nvidia/nemotron-3-ultra-550b-a55b",
    )
    NVIDIA_NIM_TEMPERATURE: float = Field(default=1.0)
    NVIDIA_NIM_TOP_P: float = Field(default=0.95)
    NVIDIA_NIM_MAX_TOKENS: int = Field(default=16384)
    NVIDIA_NIM_REASONING_BUDGET: int = Field(default=16384)
    NVIDIA_NIM_TIMEOUT_SECS: float = Field(default=600.0)
    NVIDIA_NIM_STREAM: bool = Field(default=True)

    # LiveKit real-time communication.
    LIVEKIT_URL: str = Field(default="")
    LIVEKIT_API_KEY: str = Field(default="")
    LIVEKIT_API_SECRET: str = Field(default="")
    LIVEKIT_AGENT_NAME: str = Field(default="interview-agent")

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "core_api"
    POSTGRES_PASSWORD: str = "core_api_password"
    POSTGRES_DB: str = "core_api_db"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None
    CELERY_BROKER_URL: str | None = None
    CELERY_RESULT_BACKEND: str | None = None

    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 10
    DB_POOL_RECYCLE: int = 3600
    DB_CONNECT_TIMEOUT: int = 180
    DB_COMMAND_TIMEOUT: int = 2400
    DB_STATEMENT_TIMEOUT_MS: int = 2_400_000

    REDIS_SOCKET_CONNECT_TIMEOUT: int = 5
    REDIS_SOCKET_TIMEOUT: int = 5
    REDIS_HEALTHCHECK_INTERVAL: int = 30

    @property
    def DATABASE_URL(self) -> str:
        """Build the escaped async PostgreSQL connection URL."""

        return (
            f"postgresql+asyncpg://{quote_plus(self.POSTGRES_USER)}"
            f":{quote_plus(self.POSTGRES_PASSWORD)}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def REDIS_URL(self) -> str:
        """Build the escaped Redis connection URL."""

        password = f":{quote_plus(self.REDIS_PASSWORD)}@" if self.REDIS_PASSWORD else ""
        return f"redis://{password}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    @property
    def celery_broker_url(self) -> str:
        """Return the explicit Celery broker or the shared Redis URL."""

        return self.CELERY_BROKER_URL or self.REDIS_URL

    @property
    def celery_result_backend(self) -> str:
        """Return the explicit result backend or the shared Redis URL."""

        return self.CELERY_RESULT_BACKEND or self.REDIS_URL


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process."""

    return Settings()


settings = get_settings()
