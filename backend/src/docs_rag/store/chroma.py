"""ChromaDB implementation of the vector store."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI

from docs_rag.embeddings import Vector
from docs_rag.ingestion import Chunk
from docs_rag.store.base import SearchResult, VectorStoreError

DEFAULT_COLLECTION = "docs-rag-chunks"


def chunk_id(document_id: str, index: int) -> str:
    return f"{document_id}:{index}"


class ChromaVectorStore:
    """Vector store backed by a ChromaDB collection using cosine distance.

    Embeddings are always computed by our own providers and passed in, so
    Chroma's built-in embedding function is disabled. The collection records
    which embedder produced its vectors: mixing vectors from different
    models would make similarity scores meaningless, so a mismatch fails.
    """

    def __init__(
        self,
        client: ClientAPI,
        embedder_name: str,
        collection_name: str = DEFAULT_COLLECTION,
    ) -> None:
        self._collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embedder": embedder_name},
            embedding_function=None,
        )
        stored = (self._collection.metadata or {}).get("embedder")
        if stored != embedder_name:
            raise VectorStoreError(
                f"Collection '{collection_name}' was built with embedder '{stored}', "
                f"but '{embedder_name}' is configured. Re-index the documents or use "
                "a different data directory."
            )

    @classmethod
    def persistent(cls, path: Path, embedder_name: str) -> "ChromaVectorStore":
        """Open (or create) a store persisted on disk at ``path``."""
        path.mkdir(parents=True, exist_ok=True)
        return cls(chromadb.PersistentClient(path=str(path)), embedder_name)

    def add_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Vector],
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        vectors: list[Sequence[float]] = [list(vector) for vector in embeddings]
        self._collection.add(
            ids=[chunk_id(document_id, chunk.index) for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "document_id": document_id,
                    "filename": filename,
                    "page": chunk.page,
                    "chunk_index": chunk.index,
                    "start": chunk.start,
                }
                for chunk in chunks
            ],
        )

    def search(
        self,
        query_embedding: Vector,
        top_k: int,
        document_ids: Sequence[str] | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0 or self._collection.count() == 0:
            return []

        where: dict[str, Any] | None = None
        if document_ids is not None:
            if not document_ids:
                return []
            where = {"document_id": {"$in": list(document_ids)}}

        queries: list[Sequence[float]] = [list(query_embedding)]
        result = self._collection.query(
            query_embeddings=queries,
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = (result["documents"] or [[]])[0]
        metadatas = (result["metadatas"] or [[]])[0]
        distances = (result["distances"] or [[]])[0]

        return [
            SearchResult(
                chunk_id=id_,
                document_id=str(metadata["document_id"]),
                filename=str(metadata["filename"]),
                page=int(str(metadata["page"])),
                chunk_index=int(str(metadata["chunk_index"])),
                text=text,
                # Chroma's cosine distance is 1 - cosine similarity.
                score=round(1.0 - distance, 6),
            )
            for id_, text, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]

    def delete_document(self, document_id: str) -> int:
        existing = self._collection.get(where={"document_id": document_id}, include=[])
        if existing["ids"]:
            self._collection.delete(ids=existing["ids"])
        return len(existing["ids"])

    def count(self, document_id: str | None = None) -> int:
        if document_id is None:
            return self._collection.count()
        return len(self._collection.get(where={"document_id": document_id}, include=[])["ids"])
