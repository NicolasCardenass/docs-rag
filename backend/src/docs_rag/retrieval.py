"""Semantic search over the indexed chunks."""

from collections.abc import Sequence

from docs_rag.embeddings import Embedder
from docs_rag.store import SearchResult, VectorStore


class Retriever:
    """Finds the chunks most relevant to a natural-language query."""

    def __init__(self, embedder: Embedder, store: VectorStore, default_top_k: int) -> None:
        self._embedder = embedder
        self._store = store
        self.default_top_k = default_top_k

    @property
    def embedder_name(self) -> str:
        return self._embedder.name

    def search(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        """Embed the query and return the closest chunks, best match first."""
        query = query.strip()
        if not query:
            return []
        embedding = self._embedder.embed_query(query)
        return self._store.search(embedding, top_k or self.default_top_k, document_ids)
