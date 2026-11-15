"""Document management: metadata repository and ingestion service."""

from docs_rag.documents.repository import DocumentRecord, DocumentRepository
from docs_rag.documents.service import DocumentService, DocumentTooLargeError

__all__ = ["DocumentRecord", "DocumentRepository", "DocumentService", "DocumentTooLargeError"]
