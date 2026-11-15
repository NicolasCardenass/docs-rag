"""Deterministic, dependency-free embeddings for development and tests."""

import hashlib
import math
import re
from collections.abc import Sequence

from docs_rag.embeddings.base import Vector

_TOKEN = re.compile(r"\w+", re.UNICODE)


class MockEmbedder:
    """Bag-of-words embeddings using the hashing trick.

    Each lowercase token is hashed into one of ``dimension`` buckets with a
    sign, and the resulting vector is L2-normalized. Texts sharing words get
    a higher cosine similarity, which is enough to exercise the whole
    retrieval pipeline without a model or API key. It does not capture
    synonyms or meaning beyond word overlap.
    """

    def __init__(self, dimension: int = 256) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    @property
    def name(self) -> str:
        return f"mock:hashing-{self.dimension}"

    def _embed(self, text: str) -> Vector:
        vector = [0.0] * self.dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            bucket = value % self.dimension
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(component * component for component in vector))
        return [component / norm for component in vector] if norm else vector

    def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)
