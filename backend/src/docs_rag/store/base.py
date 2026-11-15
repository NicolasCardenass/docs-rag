"""Vector store interface used by ingestion and retrieval."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from docs_rag.embeddings import Vector
from docs_rag.ingestion import Chunk


class VectorStoreError(Exception):
    """Raised when the vector store cannot be used as configured."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    score: float
    """Cosine similarity between query and chunk, from -1 (opposite) to 1 (identical)."""


class VectorStore(Protocol):
    """Stores chunk embeddings and finds the chunks closest to a query vector."""

    def add_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Vector],
    ) -> None: ...

    def search(
        self,
        query_embedding: Vector,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[SearchResult]: ...

    def delete_document(self, document_id: str) -> int:
        """Delete every chunk of a document and return how many were removed."""
        ...

    def count(self, document_id: str | None = None) -> int: ...
