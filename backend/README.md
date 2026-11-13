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
