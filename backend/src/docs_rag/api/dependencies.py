"""FastAPI dependencies that expose the services stored on the app."""

from typing import Annotated

from fastapi import Depends, Request

from docs_rag.chat import ChatService
from docs_rag.documents import DocumentService
from docs_rag.retrieval import Retriever


def get_document_service(request: Request) -> DocumentService:
    service: DocumentService = request.app.state.document_service
    return service


def get_retriever(request: Request) -> Retriever:
    retriever: Retriever = request.app.state.retriever
    return retriever


def get_chat_service(request: Request) -> ChatService:
    service: ChatService = request.app.state.chat_service
    return service


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
ChatServiceDep = Annotated[ChatService, Depends(get_chat_service)]
