import json
from collections.abc import Iterator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from docs_rag.api.dependencies import ChatServiceDep
from docs_rag.chat import Source, cited_numbers, make_sources
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


def sse_event(event: str, data: object) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post(
    "/chat/stream",
    summary="Ask a question and stream the answer (Server-Sent Events)",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "Events: `sources` (list of passages), `token` ({text}) repeated, "
                "then `done` ({answer, cited, model}) or `error` ({detail})."
            ),
        },
        502: {},
    },
)
def chat_stream(request: ChatRequest, service: ChatServiceDep) -> StreamingResponse:
    """Like `POST /chat`, but the answer is sent token by token as it is generated."""
    try:
        results = service.retrieve(request.question, request.top_k, request.document_ids)
    except EmbeddingError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    def events() -> Iterator[str]:
        sources = [SourceOut.from_source(s) for s in make_sources(results, answer="")]
        yield sse_event("sources", [source.model_dump(exclude={"cited"}) for source in sources])
        parts: list[str] = []
        try:
            for text in service.stream(request.question, results):
                parts.append(text)
                yield sse_event("token", {"text": text})
        except LLMError as exc:
            yield sse_event("error", {"detail": str(exc)})
            return
        answer = "".join(parts)
        yield sse_event(
            "done",
            {
                "answer": answer,
                "cited": sorted(cited_numbers(answer, len(results))),
                "model": service.model_name,
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Disable proxy buffering so tokens reach the browser immediately.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
