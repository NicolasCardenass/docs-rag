from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docs_rag.config import Settings
from docs_rag.main import create_app

TEXT = (
    "Docs RAG indexes documents for question answering.\n\n"
    "Each document is split into chunks, embedded and stored in a vector database."
)


def upload(client: TestClient, filename: str, content: bytes) -> dict[str, object]:
    response = client.post("/documents", files={"file": (filename, content)})
    assert response.status_code == 201, response.text
    body: dict[str, object] = response.json()
    return body


def test_upload_indexes_document(client: TestClient) -> None:
    body = upload(client, "intro.md", TEXT.encode())

    assert body["filename"] == "intro.md"
    assert body["media_type"] == "text/markdown"
    assert body["pages"] == 1
    assert isinstance(body["chunks"], int)
    assert body["chunks"] >= 1
    assert body["size_bytes"] == len(TEXT.encode())
    assert isinstance(body["id"], str)


def test_list_get_and_delete(client: TestClient) -> None:
    first = upload(client, "a.txt", b"First document.")
    second = upload(client, "b.txt", b"Second document.")

    listed = client.get("/documents").json()
    assert [doc["id"] for doc in listed] == [second["id"], first["id"]]  # newest first

    assert client.get(f"/documents/{first['id']}").json()["filename"] == "a.txt"

    assert client.delete(f"/documents/{first['id']}").status_code == 204
    assert client.get(f"/documents/{first['id']}").status_code == 404
    assert client.delete(f"/documents/{first['id']}").status_code == 404
    assert [doc["id"] for doc in client.get("/documents").json()] == [second["id"]]


def test_deleting_a_document_removes_its_chunks(client: TestClient) -> None:
    document = upload(client, "a.txt", TEXT.encode())
    service = client.app.state.document_service  # type: ignore[attr-defined]
    store = service._store

    assert store.count(document["id"]) == document["chunks"]
    client.delete(f"/documents/{document['id']}")
    assert store.count(document["id"]) == 0


@pytest.mark.parametrize(
    ("filename", "content", "status", "detail"),
    [
        ("slides.pptx", b"data", 415, "Unsupported file type"),
        ("empty.txt", b"   ", 422, "No extractable text"),
        ("broken.pdf", b"%PDF-1.7 not a pdf", 422, "Could not read PDF"),
    ],
)
def test_upload_errors(
    client: TestClient, filename: str, content: bytes, status: int, detail: str
) -> None:
    response = client.post("/documents", files={"file": (filename, content)})

    assert response.status_code == status
    assert detail in response.json()["detail"]
    assert client.get("/documents").json() == []


def test_rejects_files_over_the_size_limit(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, data_dir=tmp_path, max_upload_mb=1)  # type: ignore[call-arg]
    with TestClient(create_app(settings)) as client:
        response = client.post("/documents", files={"file": ("big.txt", b"a " * 600_000)})

    assert response.status_code == 413


def test_documents_persist_across_restarts(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        document = upload(client, "a.txt", TEXT.encode())

    with TestClient(create_app(settings)) as client:
        assert [doc["id"] for doc in client.get("/documents").json()] == [document["id"]]
