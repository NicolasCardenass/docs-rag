"""Embeddings from a local Ollama server."""

from collections.abc import Sequence
from typing import Any

import httpx2

from docs_rag.embeddings.base import EmbeddingError, Vector


class OllamaEmbedder:
    """Embeds text with a model served by Ollama (``POST /api/embed``)."""

    def __init__(self, client: httpx2.Client, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        if not texts:
            return []
        try:
            response = self._client.post(
                "/api/embed", json={"model": self._model, "input": list(texts)}
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except httpx2.HTTPError as exc:
            raise EmbeddingError(f"Ollama embeddings request failed: {exc}") from exc

        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise EmbeddingError("Ollama returned an unexpected embeddings payload")
        return [[float(value) for value in vector] for vector in embeddings]

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]
