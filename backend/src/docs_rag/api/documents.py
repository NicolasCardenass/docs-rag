from fastapi import APIRouter, HTTPException, Response, UploadFile, status

from docs_rag.api.dependencies import DocumentServiceDep
from docs_rag.documents import DocumentRecord, DocumentTooLargeError
from docs_rag.embeddings import EmbeddingError
from docs_rag.ingestion import ExtractionError, UnsupportedFileTypeError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a document",
    responses={
        413: {"description": "File too large"},
        415: {"description": "Unsupported file type"},
        422: {"description": "No extractable text or unreadable file"},
        502: {"description": "Embedding provider failed"},
    },
)
async def upload_document(file: UploadFile, service: DocumentServiceDep) -> DocumentRecord:
    """Extract, chunk and embed a PDF, TXT or Markdown file."""
    data = await file.read()
    filename = file.filename or "document"
    try:
        return service.ingest(filename, data)
    except DocumentTooLargeError as exc:
        raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, str(exc)) from exc
    except UnsupportedFileTypeError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc
    except ExtractionError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc


@router.get("", summary="List indexed documents")
def list_documents(service: DocumentServiceDep) -> list[DocumentRecord]:
    return service.list()


@router.get("/{document_id}", summary="Get a document", responses={404: {}})
def get_document(document_id: str, service: DocumentServiceDep) -> DocumentRecord:
    record = service.get(document_id)
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return record


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its chunks",
    responses={404: {}},
)
def delete_document(document_id: str, service: DocumentServiceDep) -> Response:
    if not service.delete(document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
