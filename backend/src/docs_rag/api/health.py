from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from docs_rag import __version__

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str


@router.get("/health", summary="Liveness check")
def health() -> HealthResponse:
    """Report that the API process is up and which version is running."""
    return HealthResponse(status="ok", version=__version__)
