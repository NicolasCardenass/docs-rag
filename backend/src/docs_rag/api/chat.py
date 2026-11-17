from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from docs_rag.api.dependencies import ChatServiceDep
from docs_rag.chat import Source
from docs_rag.embeddings import EmbeddingError
from docs_rag.llm import LLMError


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["What does the manual say?"])
    top_k: int | None = Field(default=None, ge=1, le=20, description="Defaults to DOCS_RAG_TOP_K")
    document_ids: list[str] | None = Field(
        default=None, description="Restrict the answer to these documents"
    )


class SourceOut(BaseModel):
    number: int
    document_id: str
    filename: str
    page: int
    text: str
    score: float
    cited: bool

    @classmethod
    def from_source(cls, source: Source) -> "SourceOut":
        result = source.result
        return cls(
            number=source.number,
            document_id=result.document_id,
            filename=result.filename,
            page=result.page,
            text=result.text,
            score=result.score,
            cited=source.cited,
        )


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceOut]
    model: str


router = APIRouter(tags=["chat"])


@router.post("/chat", summary="Ask a question about the documents", responses={502: {}})
def chat(request: ChatRequest, service: ChatServiceDep) -> ChatResponse:
    """Answer with retrieval-augmented generation, citing passages as [n]."""
    try:
        result = service.answer(request.question, request.top_k, request.document_ids)
    except (EmbeddingError, LLMError) as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return ChatResponse(
        answer=result.answer,
        sources=[SourceOut.from_source(source) for source in result.sources],
        model=result.model,
    )
