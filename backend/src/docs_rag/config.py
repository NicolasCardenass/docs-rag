"""Application settings loaded from environment variables (and an optional ``.env`` file)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EmbeddingProvider = Literal["mock", "openai", "ollama"]
LLMProvider = Literal["mock", "openai", "anthropic", "ollama"]


class Settings(BaseSettings):
    """Runtime configuration.

    Every field can be set with an environment variable prefixed with
    ``DOCS_RAG_`` (e.g. ``DOCS_RAG_LLM_PROVIDER=openai``).
    """

    model_config = SettingsConfigDict(
        env_prefix="DOCS_RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # General
    environment: Literal["development", "production", "test"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    data_dir: Path = Path("data")
    cors_origins: list[str] = ["http://localhost:3000"]

    # Ingestion
    max_upload_mb: int = Field(default=20, gt=0, le=200)
    chunk_size: int = Field(default=800, ge=100, le=8000, description="Characters per chunk")
    chunk_overlap: int = Field(default=120, ge=0, description="Characters shared by chunks")

    # Retrieval
    top_k: int = Field(default=4, ge=1, le=20)

    # Providers
    embedding_provider: EmbeddingProvider = "mock"
    llm_provider: LLMProvider = "mock"
    openai_api_key: SecretStr | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-haiku-4-5"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "nomic-embed-text"
    ollama_chat_model: str = "llama3.2"

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        uses_openai = "openai" in (self.embedding_provider, self.llm_provider)
        if uses_openai and self.openai_api_key is None:
            raise ValueError("DOCS_RAG_OPENAI_API_KEY is required when using the OpenAI provider")
        if self.llm_provider == "anthropic" and self.anthropic_api_key is None:
            raise ValueError(
                "DOCS_RAG_ANTHROPIC_API_KEY is required when using the Anthropic provider"
            )
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, read once and cached.

    Used as a FastAPI dependency, so tests can override it.
    """
    return Settings()
