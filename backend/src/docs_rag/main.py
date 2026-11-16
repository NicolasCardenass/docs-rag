"""Application entry point.

Run locally with::

    uv run uvicorn docs_rag.main:app --reload
"""

from dataclasses import dataclass

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docs_rag import __version__
from docs_rag.api import documents, health, search
from docs_rag.config import Settings, get_settings
from docs_rag.documents import DocumentRepository, DocumentService
from docs_rag.embeddings import create_embedder
from docs_rag.retrieval import Retriever
from docs_rag.store import ChromaVectorStore


@dataclass(frozen=True)
class Services:
    documents: DocumentService
    retriever: Retriever


def build_services(settings: Settings) -> Services:
    """Wire the ingestion and retrieval pipelines from settings.

    Both share the same embedder and vector store, so documents are
    embedded and searched with the same model.
    """
    embedder = create_embedder(settings)
    store = ChromaVectorStore.persistent(settings.data_dir / "index", embedder.name)
    repository = DocumentRepository(settings.data_dir / "documents.json")
    documents = DocumentService(
        repository,
        store,
        embedder,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        max_upload_bytes=settings.max_upload_bytes,
    )
    return Services(documents=documents, retriever=Retriever(embedder, store, settings.top_k))


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
    services = build_services(settings)
    app.state.document_service = services.documents
    app.state.retriever = services.retriever
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(search.router)
    return app


app = create_app()
