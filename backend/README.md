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

## API

| Method   | Path              | Description                                                            |
| -------- | ----------------- | ---------------------------------------------------------------------- |
| `GET`    | `/health`         | Liveness check with the running version                                |
| `POST`   | `/documents`      | Upload a PDF, TXT or Markdown file (multipart field `file`)            |
| `GET`    | `/documents`      | List indexed documents, newest first                                   |
| `GET`    | `/documents/{id}` | Get one document's metadata                                            |
| `POST`   | `/search`         | Semantic search: `{"query": "...", "top_k": 4, "document_ids": [...]}` |
| `POST`   | `/chat`           | Ask a question: answer with `[n]` citations and the source passages    |
| `POST`   | `/chat/stream`    | Same as `/chat`, streamed as Server-Sent Events                        |
| `DELETE` | `/documents/{id}` | Delete a document and all of its chunks                                |

Example:

```bash
curl -F file=@manual.pdf http://localhost:8000/documents
```

Uploaded files are not stored: their text is chunked and embedded, chunks go to a ChromaDB index
in `DOCS_RAG_DATA_DIR/index` and document metadata to `DOCS_RAG_DATA_DIR/documents.json`.

### Streaming answers

`POST /chat/stream` sends [Server-Sent Events](https://developer.mozilla.org/docs/Web/API/Server-sent_events)
so the answer appears while it is being generated:

| Event     | Data                                                                  |
| --------- | --------------------------------------------------------------------- |
| `sources` | The numbered passages used as context (sent first)                    |
| `token`   | `{"text": "..."}`, a fragment of the answer (repeated)                |
| `done`    | `{"answer": "...", "cited": [1, 2], "model": "..."}`                  |
| `error`   | `{"detail": "..."}` if the provider fails after streaming has started |

```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What do inverters do?"}'
```
