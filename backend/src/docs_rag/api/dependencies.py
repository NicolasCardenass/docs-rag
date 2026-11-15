"""FastAPI dependencies that expose the services stored on the app."""

from typing import Annotated

from fastapi import Depends, Request

from docs_rag.documents import DocumentService


def get_document_service(request: Request) -> DocumentService:
    service: DocumentService = request.app.state.document_service
    return service


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
