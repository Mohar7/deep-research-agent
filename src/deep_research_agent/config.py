"""Runtime configuration, loaded from env / `.env`."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the research agent."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- Model ----
    # `provider:model` syntax for init_chat_model. Examples:
    #   openai:gpt-4o-mini
    #   anthropic:claude-haiku-4-5
    model_name: str = "openai:gpt-4o-mini"
    model_temperature: float = 0.0

    # ---- Research budget ----
    # Hard cap on planner → researcher loops. Stops runaway costs.
    max_iterations: int = 3
    # Number of web results to fetch per subquery.
    results_per_query: int = 5

    # ---- Persistence ----
    sqlite_path: str = "./checkpoints.sqlite"

    # ---- API keys (loaded from env, not stored here) ----
    # OPENAI_API_KEY / ANTHROPIC_API_KEY / LANGFUSE_PUBLIC_KEY etc.
    # are read by their respective libraries from os.environ.

    # ---- Observability ----
    enable_langfuse: bool = False  # opt-in via env LANGFUSE_ENABLED=true


@lru_cache
def get_settings() -> Settings:
    return Settings()
