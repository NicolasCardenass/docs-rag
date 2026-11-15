"""Application entry point.

Run locally with::

    uv run uvicorn docs_rag.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docs_rag import __version__
from docs_rag.api import documents, health
from docs_rag.config import Settings, get_settings
from docs_rag.documents import DocumentRepository, DocumentService
from docs_rag.embeddings import create_embedder
from docs_rag.store import ChromaVectorStore


def build_document_service(settings: Settings) -> DocumentService:
    """Wire the ingestion pipeline from settings."""
    embedder = create_embedder(settings)
    store = ChromaVectorStore.persistent(settings.data_dir / "index", embedder.name)
    repository = DocumentRepository(settings.data_dir / "documents.json")
    return DocumentService(
        repository,
        store,
        embedder,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_bytes=settings.max_upload_bytes,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    A factory keeps app creation explicit and lets tests build fresh,
    isolated instances with their own settings.
    """
    settings = settings or get_settings()

    app = FastAPI(
        title="Docs RAG API",
        version=__version__,
        description=(
            "Upload documents and ask questions answered with retrieval-augmented generation."
        ),
    )
    app.state.settings = settings
    app.state.document_service = build_document_service(settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    return app


app = create_app()
