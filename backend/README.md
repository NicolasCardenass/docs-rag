# Docs RAG — backend

Python API that ingests documents, indexes them as embeddings and answers questions with an LLM,
citing the passages it used.

## Development

Requirements: Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cd backend
uv sync              # create .venv and install dependencies
uv run pytest        # tests
uv run ruff check .  # lint
uv run ruff format . # format
uv run mypy          # type-check
```

## Configuration

Settings are read from environment variables prefixed with `DOCS_RAG_` or from a `backend/.env`
file. Copy [`.env.example`](./.env.example) to `.env` to start; every value is optional. By default
the app uses deterministic **mock** embedding and LLM providers, so it runs without API keys.
Selecting a real provider without its API key fails fast at startup with a clear error.

## Running the API

```bash
uv run uvicorn docs_rag.main:app --reload
```

The API listens on <http://localhost:8000>. Interactive documentation is available at
<http://localhost:8000/docs> (Swagger UI) and the health check at `GET /health`.
