"""Application entry point.

Run locally with::

    uv run uvicorn docs_rag.main:app --reload
"""

from fastapi import FastAPI

from docs_rag import __version__
from docs_rag.api import health


def create_app() -> FastAPI:
    """Build the FastAPI application.

    A factory keeps app creation explicit and lets tests build fresh,
    isolated instances.
    """
    app = FastAPI(
        title="Docs RAG API",
        version=__version__,
        description=(
            "Upload documents and ask questions answered with retrieval-augmented generation."
        ),
    )
    app.include_router(health.router)
    return app


app = create_app()
