"""Application settings for interview-engine-service."""

from functools import lru_cache
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env",),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "interview-engine-service"
    APP_ENV: str = Field(default="development")

    # ── Deepgram (STT + TTS) ──────────────────────────────────────────────────
    DEEPGRAM_API_KEY: str = Field(default="")
    DEEPGRAM_STT_MODEL: str = Field(default="nova-2")
    DEEPGRAM_TTS_MODEL: str = Field(default="aura-asteria-en")

    # ── Groq (LLM) ───────────────────────────────────────────────────────────
    GROQ_API_KEY: str = Field(default="")
    GROQ_MODEL: str = Field(default="llama-3.1-8b-instant")
    GROQ_MAX_TOKENS: int = Field(default=512)
    GROQ_TEMPERATURE: float = Field(default=0.7)

    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "core_api"
    POSTGRES_PASSWORD: str = "core_api_password"
    POSTGRES_DB: str = "core_api_db"

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None

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
        return (
            f"postgresql+asyncpg://{quote_plus(self.POSTGRES_USER)}"
            f":{quote_plus(self.POSTGRES_PASSWORD)}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def REDIS_URL(self) -> str:
        password = f":{quote_plus(self.REDIS_PASSWORD)}@" if self.REDIS_PASSWORD else ""
        return f"redis://{password}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
