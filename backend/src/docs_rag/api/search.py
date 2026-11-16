from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from docs_rag.api.dependencies import RetrieverDep
from docs_rag.embeddings import EmbeddingError


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, examples=["How is the text chunked?"])
    top_k: int | None = Field(default=None, ge=1, le=20, description="Defaults to DOCS_RAG_TOP_K")
    document_ids: list[str] | None = Field(
        default=None, description="Restrict the search to these documents"
    )


class SearchHit(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHit]


router = APIRouter(tags=["search"])


@router.post("/search", summary="Semantic search over indexed chunks", responses={502: {}})
def search(request: SearchRequest, retriever: RetrieverDep) -> SearchResponse:
    """Return the chunks most similar to the query, best match first."""
    try:
        results = retriever.search(request.query, request.top_k, request.document_ids)
    except EmbeddingError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    return SearchResponse(
        query=request.query,
        results=[SearchHit.model_validate(result, from_attributes=True) for result in results],
    )
