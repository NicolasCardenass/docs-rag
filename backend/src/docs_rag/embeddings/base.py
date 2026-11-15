"""Embedding provider interface."""

from collections.abc import Sequence
from typing import Protocol

Vector = list[float]


class EmbeddingError(Exception):
    """Raised when a provider fails to compute embeddings."""


class Embedder(Protocol):
    """Turns text into fixed-size vectors whose distance reflects meaning.

    Documents and queries have separate methods because some models embed
    them differently (asymmetric retrieval).
    """

    @property
    def name(self) -> str:
        """Provider and model, e.g. ``"openai:text-embedding-3-small"``."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...
