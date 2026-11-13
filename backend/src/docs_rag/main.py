"""Application entry point.

Run locally with::

    uv run uvicorn docs_rag.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docs_rag import __version__
from docs_rag.api import health
from docs_rag.config import Settings, get_settings


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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    return app


app = create_app()
