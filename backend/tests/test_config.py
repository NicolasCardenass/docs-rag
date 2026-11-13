import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from docs_rag.config import Settings


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Ignore the developer's environment and any local .env file."""
    for key in list(os.environ):
        if key.startswith("DOCS_RAG_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)


def test_defaults_run_without_any_api_key() -> None:
    settings = Settings()

    assert settings.embedding_provider == "mock"
    assert settings.llm_provider == "mock"
    assert settings.chunk_overlap < settings.chunk_size
    assert settings.max_upload_bytes == 20 * 1024 * 1024


def test_reads_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOCS_RAG_TOP_K", "8")
    monkeypatch.setenv("DOCS_RAG_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("DOCS_RAG_CORS_ORIGINS", '["https://example.com"]')

    settings = Settings()

    assert settings.top_k == 8
    assert settings.llm_provider == "ollama"
    assert settings.cors_origins == ["https://example.com"]


def test_reads_dotenv_file(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("DOCS_RAG_CHUNK_SIZE=1000\n", encoding="utf-8")

    assert Settings().chunk_size == 1000


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap must be smaller"):
        Settings(chunk_size=200, chunk_overlap=200)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"llm_provider": "openai"}, "OPENAI_API_KEY is required"),
        ({"embedding_provider": "openai"}, "OPENAI_API_KEY is required"),
        ({"llm_provider": "anthropic"}, "ANTHROPIC_API_KEY is required"),
    ],
)
def test_real_providers_require_api_keys(overrides: dict[str, str], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Settings(**overrides)  # type: ignore[arg-type]


def test_api_keys_are_not_leaked_in_repr() -> None:
    settings = Settings(llm_provider="openai", openai_api_key="sk-secret-value")  # type: ignore[arg-type]

    assert "sk-secret-value" not in repr(settings)
    assert settings.openai_api_key is not None
    assert settings.openai_api_key.get_secret_value() == "sk-secret-value"


def test_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError):
        Settings(llm_provider="unknown")  # type: ignore[arg-type]


def test_env_example_is_valid(tmp_path: Path) -> None:
    example = Path(__file__).parents[1] / ".env.example"
    (tmp_path / ".env").write_text(example.read_text(encoding="utf-8"), encoding="utf-8")

    settings = Settings()

    assert settings.llm_provider == "mock"
    assert settings.chunk_size == 800
    assert settings.cors_origins == ["http://localhost:3000"]
