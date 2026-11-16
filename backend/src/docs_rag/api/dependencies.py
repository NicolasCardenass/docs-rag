"""FastAPI dependencies that expose the services stored on the app."""

from typing import Annotated

from fastapi import Depends, Request

from docs_rag.documents import DocumentService
from docs_rag.retrieval import Retriever


def get_document_service(request: Request) -> DocumentService:
    service: DocumentService = request.app.state.document_service
    return service


def get_retriever(request: Request) -> Retriever:
    retriever: Retriever = request.app.state.retriever
    return retriever


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
