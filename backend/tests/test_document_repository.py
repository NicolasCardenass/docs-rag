from datetime import UTC, datetime, timedelta
from pathlib import Path

from docs_rag.documents import DocumentRecord, DocumentRepository


def record(id_: str, minutes: int = 0) -> DocumentRecord:
    return DocumentRecord(
        id=id_,
        filename=f"{id_}.txt",
        media_type="text/plain",
        size_bytes=10,
        pages=1,
        chunks=1,
        created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=minutes),
    )


def test_add_list_get_delete_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "documents.json"
    repository = DocumentRepository(path)
    repository.add(record("old", 0))
    repository.add(record("new", 5))

    assert [r.id for r in repository.list()] == ["new", "old"]
    assert repository.get("old") == record("old", 0)
    assert repository.delete("old") is True
    assert repository.delete("old") is False

    reloaded = DocumentRepository(path)
    assert [r.id for r in reloaded.list()] == ["new"]
    assert not path.with_suffix(".tmp").exists()
