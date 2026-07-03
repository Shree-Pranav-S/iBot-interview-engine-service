"""Application settings for interview-engine-service."""

from functools import lru_cache
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed service configuration."""

    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
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
    DEEPGRAM_ENDPOINTING_MS: int = Field(default=400)
    VAD_MIN_SPEECH_DURATION_SECS: float = Field(default=0.2)
    VAD_MIN_SILENCE_DURATION_SECS: float = Field(default=0.4)
    VAD_PREFIX_PADDING_DURATION_SECS: float = Field(default=0.15)
    # Endpointing tuned for fast, human-like turn-taking. LiveKit TurnDetector
    # gates semantic EOT; these delays only cap post-semantic silence padding.
    TURN_ENDPOINTING_MIN_DELAY_SECS: float = Field(default=0.5)
    TURN_ENDPOINTING_MAX_DELAY_SECS: float = Field(default=2.0)
    FALSE_INTERRUPTION_TIMEOUT_SECS: float = Field(default=1.2)
    # Speak an instant, content-neutral acknowledgement while the merged
    # interviewer call runs so the candidate hears a reply with no dead air.
    ENABLE_ACK_FILLER: bool = Field(default=True)
    # Start merged LLM reasoning before turn confirmation (custom llm_node wiring).
    PREEMPTIVE_GENERATION_ENABLED: bool = Field(default=False)

    # Groq models used by latency-sensitive graph nodes.
    # Existing healthy source variables are retained so local development can
    # populate the common pool without duplicating credentials.
    GROQ_API_KEY: str = Field(default="")
    FALLBACK_GROQ_API_KEY: str = Field(default="")
    GROQ_QUESTION_API_KEY: str = Field(default="")
    FALLBACK_GROQ_EVALUATION_KEY: str = Field(default="")
    # Common turn-robin pool for every in-interview Groq call.
    GROQ_INTERVIEW_API_KEY_1: str = Field(default="")
    GROQ_INTERVIEW_API_KEY_2: str = Field(default="")
    GROQ_INTERVIEW_API_KEY_3: str = Field(default="")
    GROQ_INTERVIEW_API_KEY_4: str = Field(default="")
    GROQ_QUESTION_MODEL: str = Field(default="openai/gpt-oss-120b")
    GROQ_QUESTION_MAX_TOKENS: int = Field(default=256)
    GROQ_QUESTION_TEMPERATURE: float = Field(default=0.15)
    GROQ_QUESTION_TIMEOUT_SECS: float = Field(default=10.0)
    # Single merged "interviewer turn" call: classify + evaluate + generate.
    # Reuses the question-generation API keys (same operational purpose).
    GROQ_INTERVIEWER_MODEL: str = Field(default="openai/gpt-oss-120b")
    GROQ_INTERVIEWER_MAX_TOKENS: int = Field(default=520)
    GROQ_INTERVIEWER_TEMPERATURE: float = Field(default=0.28)
    GROQ_INTERVIEWER_TIMEOUT_SECS: float = Field(default=12.0)
    GROQ_LIVE_EVAL_MODEL: str = Field(default="openai/gpt-oss-20b")
    GROQ_LIVE_EVAL_MAX_TOKENS: int = Field(default=128)
    GROQ_LIVE_EVAL_TEMPERATURE: float = Field(default=0.0)
    GROQ_LIVE_EVAL_TIMEOUT_SECS: float = Field(default=6.0)
    GROQ_CLASSIFY_MODEL: str = Field(default="openai/gpt-oss-20b")
    GROQ_CLASSIFY_MAX_TOKENS: int = Field(default=128)
    GROQ_CLASSIFY_TEMPERATURE: float = Field(default=0.0)
    GROQ_CLASSIFY_TIMEOUT_SECS: float = Field(default=5.0)
    GROQ_REASONING_EFFORT: Literal["low", "medium", "high"] = Field(default="low")
    GROQ_KEY_RATE_LIMIT_COOLDOWN_SECS: float = Field(default=60.0)
    GROQ_KEY_TRANSIENT_COOLDOWN_SECS: float = Field(default=5.0)

    # NVIDIA NIM structured holistic evaluation.
    NVIDIA_NIM_API_KEY: str = Field(default="")
    FALLBACK_NVIDIA_NIM_API_KEY: str = Field(default="")
    NVIDIA_NIM_BASE_URL: str = Field(
        default="https://integrate.api.nvidia.com/v1",
    )
    NVIDIA_NIM_MODEL: str = Field(
        default="nvidia/nemotron-3-ultra-550b-a55b",
    )
    NVIDIA_NIM_TEMPERATURE: float = Field(default=0.2)
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
    LIVEKIT_FORCE_RELAY: bool = Field(default=False)

    CORE_API_URL: str = Field(default="http://localhost:8000")
    CHECKPOINT_POSTGRES_HOST: str | None = None
    CHECKPOINT_POSTGRES_PORT: int | None = None
    CHECKPOINT_POSTGRES_USER: str | None = None
    CHECKPOINT_POSTGRES_PASSWORD: str | None = None
    CHECKPOINT_POSTGRES_DB: str | None = None

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
        """Build the escaped async PostgreSQL connection URL for LangGraph checkpoints."""

        host = self.CHECKPOINT_POSTGRES_HOST or self.POSTGRES_HOST
        port = self.CHECKPOINT_POSTGRES_PORT or self.POSTGRES_PORT
        user = self.CHECKPOINT_POSTGRES_USER or self.POSTGRES_USER
        password = self.CHECKPOINT_POSTGRES_PASSWORD or self.POSTGRES_PASSWORD
        database = self.CHECKPOINT_POSTGRES_DB or self.POSTGRES_DB
        return (
            f"postgresql+asyncpg://{quote_plus(user)}"
            f":{quote_plus(password)}"
            f"@{host}:{port}/{database}"
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
