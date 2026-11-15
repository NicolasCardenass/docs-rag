"""Document ingestion use cases: upload, list and delete."""

import uuid
from datetime import UTC, datetime

from docs_rag.documents.repository import DocumentRecord, DocumentRepository
from docs_rag.embeddings import Embedder
from docs_rag.ingestion import chunk_document, extract_text
from docs_rag.store import VectorStore


class DocumentTooLargeError(Exception):
    def __init__(self, size: int, limit: int) -> None:
        super().__init__(f"File is {size} bytes; the limit is {limit} bytes")


class DocumentService:
    """Coordinates extraction, chunking, embedding and storage of documents."""

    def __init__(
        self,
        repository: DocumentRepository,
        store: VectorStore,
        embedder: Embedder,
        *,
        chunk_size: int,
        chunk_overlap: int,
        max_upload_bytes: int,
    ) -> None:
        self._repository = repository
        self._store = store
        self._embedder = embedder
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_upload_bytes = max_upload_bytes

    def ingest(self, filename: str, data: bytes) -> DocumentRecord:
        """Index a file so it can be searched.

        Raises:
            DocumentTooLargeError: the file exceeds the configured limit.
            docs_rag.ingestion.ExtractionError: the file cannot be read.
            docs_rag.embeddings.EmbeddingError: the embedding provider failed.
        """
        if len(data) > self._max_upload_bytes:
            raise DocumentTooLargeError(len(data), self._max_upload_bytes)

        document = extract_text(filename, data)
        chunks = chunk_document(document, self._chunk_size, self._chunk_overlap)
        embeddings = self._embedder.embed_documents([chunk.text for chunk in chunks])

        record = DocumentRecord(
            id=uuid.uuid4().hex,
            filename=filename,
            media_type=document.media_type,
            size_bytes=len(data),
            pages=len(document.pages),
            chunks=len(chunks),
            created_at=datetime.now(UTC),
        )
        self._store.add_chunks(record.id, filename, chunks, embeddings)
        try:
            self._repository.add(record)
        except Exception:
            # Keep the store and the repository consistent.
            self._store.delete_document(record.id)
            raise
        return record

    def list(self) -> list[DocumentRecord]:
        return self._repository.list()

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._repository.get(document_id)

    def delete(self, document_id: str) -> bool:
        """Remove a document and its chunks. Returns False if it did not exist."""
        if self._repository.get(document_id) is None:
            return False
        self._store.delete_document(document_id)
        return self._repository.delete(document_id)
