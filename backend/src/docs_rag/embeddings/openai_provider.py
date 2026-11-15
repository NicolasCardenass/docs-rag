"""Embeddings from the OpenAI API."""

from collections.abc import Sequence

import openai

from docs_rag.embeddings.base import EmbeddingError, Vector


class OpenAIEmbedder:
    """Embeds text with an OpenAI embedding model, in batches."""

    def __init__(self, client: openai.OpenAI, model: str, batch_size: int = 128) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            try:
                response = self._client.embeddings.create(model=self._model, input=batch)
            except openai.OpenAIError as exc:
                raise EmbeddingError(f"OpenAI embeddings request failed: {exc}") from exc
            # The API returns one item per input; sort by index to be safe.
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        return vectors

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]
