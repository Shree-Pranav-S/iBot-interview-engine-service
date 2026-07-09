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
    DEEPGRAM_API_KEY: str = ""
    DEEPGRAM_STT_MODEL: str = "nova-3"
    DEEPGRAM_TTS_MODEL: str = "aura-2-andromeda-en"
    VAD_MIN_SPEECH_DURATION_SECS: float = 0.2
    VAD_MIN_SILENCE_DURATION_SECS: float = 0.4
    VAD_PREFIX_PADDING_DURATION_SECS: float = 0.15
    # Endpointing tuned for fast, human-like turn-taking. LiveKit TurnDetector
    # gates semantic EOT; these delays only cap post-semantic silence padding.
    TURN_ENDPOINTING_MIN_DELAY_SECS: float = 0.5
    TURN_ENDPOINTING_MAX_DELAY_SECS: float = 2.0
    FALSE_INTERRUPTION_TIMEOUT_SECS: float = 1.2
    # Speak an instant, content-neutral acknowledgement while live turn reasoning
    # runs so the candidate hears a reply with no dead air.
    ENABLE_ACK_FILLER: bool = True

    # Groq models used by latency-sensitive graph nodes.
    # Existing healthy source variables are retained so local development can
    # populate the common pool without duplicating credentials.
    GROQ_API_KEY: str = ""
    FALLBACK_GROQ_API_KEY: str = ""
    GROQ_QUESTION_API_KEY: str = ""
    FALLBACK_GROQ_EVALUATION_KEY: str = ""
    # Common turn-robin pool for every in-interview Groq call.
    GROQ_INTERVIEW_API_KEY_1: str = ""
    GROQ_INTERVIEW_API_KEY_2: str = ""
    GROQ_INTERVIEW_API_KEY_3: str = ""
    GROQ_INTERVIEW_API_KEY_4: str = ""
    GROQ_QUESTION_MODEL: str = "openai/gpt-oss-120b"
    GROQ_QUESTION_MAX_TOKENS: int = 256
    GROQ_QUESTION_TEMPERATURE: float = 0.15
    GROQ_QUESTION_TIMEOUT_SECS: float = 10.0
    # Stage-two live evaluation and interviewer response generation.
    GROQ_INTERVIEWER_MODEL: str = "openai/gpt-oss-120b"
    GROQ_INTERVIEWER_MAX_TOKENS: int = 520
    GROQ_INTERVIEWER_TEMPERATURE: float = 0.28
    GROQ_INTERVIEWER_TIMEOUT_SECS: float = 12.0
    # Stage-one classification is deliberately small, strict, and low variance.
    GROQ_CLASSIFY_MODEL: str = "openai/gpt-oss-20b"
    GROQ_CLASSIFY_MAX_TOKENS: int = 128
    GROQ_CLASSIFY_TEMPERATURE: float = 0.0
    GROQ_CLASSIFY_TIMEOUT_SECS: float = 5.0
    GROQ_REASONING_EFFORT: Literal["low", "medium", "high"] = "low"
    GROQ_KEY_RATE_LIMIT_COOLDOWN_SECS: float = 60.0
    GROQ_KEY_TRANSIENT_COOLDOWN_SECS: float = 5.0

    # NVIDIA NIM structured holistic evaluation.
    NVIDIA_NIM_API_KEY: str = ""
    FALLBACK_NVIDIA_NIM_API_KEY: str = ""
    NVIDIA_NIM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"
    NVIDIA_NIM_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b"
    NVIDIA_NIM_TEMPERATURE: float = 0.2
    NVIDIA_NIM_TOP_P: float = 0.95
    NVIDIA_NIM_MAX_TOKENS: int = 16384
    NVIDIA_NIM_REASONING_BUDGET: int = 16384
    NVIDIA_NIM_TIMEOUT_SECS: float = 600.0
    NVIDIA_NIM_STREAM: bool = True

    # LiveKit real-time communication.
    LIVEKIT_URL: str = ""
    LIVEKIT_API_KEY: str = ""
    LIVEKIT_API_SECRET: str = ""
    LIVEKIT_AGENT_NAME: str = "interview-agent"
    LIVEKIT_FORCE_RELAY: bool = False

    CORE_API_URL: str = "http://localhost:8000"
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
            f"postgresql://{quote_plus(user)}"
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
