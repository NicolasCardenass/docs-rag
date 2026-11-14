"""Document ingestion: text extraction and chunking."""

from docs_rag.ingestion.chunking import Chunk, chunk_document, split_text
from docs_rag.ingestion.extract import (
    SUPPORTED_EXTENSIONS,
    DocumentText,
    EmptyDocumentError,
    ExtractionError,
    PageText,
    UnsupportedFileTypeError,
    extract_text,
)

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "Chunk",
    "DocumentText",
    "EmptyDocumentError",
    "ExtractionError",
    "PageText",
    "UnsupportedFileTypeError",
    "chunk_document",
    "extract_text",
    "split_text",
]
