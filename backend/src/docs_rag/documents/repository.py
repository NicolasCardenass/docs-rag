"""Persistence of document metadata (the chunks themselves live in the vector store)."""

import os
import threading
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, TypeAdapter


class DocumentRecord(BaseModel):
    id: str
    filename: str
    media_type: str
    size_bytes: int
    pages: int
    chunks: int
    created_at: datetime


_RECORDS = TypeAdapter(list[DocumentRecord])


class DocumentRepository:
    """Stores document records in a JSON file.

    Small and dependency-free, which is enough for a single-process app.
    Writes go to a temporary file that is then atomically renamed, so a
    crash never leaves a half-written file behind.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._records: dict[str, DocumentRecord] = {}
        if path.exists():
            loaded = _RECORDS.validate_json(path.read_bytes())
            self._records = {record.id: record for record in loaded}

    def list(self) -> list[DocumentRecord]:
        """All documents, newest first."""
        return sorted(self._records.values(), key=lambda r: r.created_at, reverse=True)

    def get(self, document_id: str) -> DocumentRecord | None:
        return self._records.get(document_id)

    def add(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.id] = record
            self._save()

    def delete(self, document_id: str) -> bool:
        with self._lock:
            if self._records.pop(document_id, None) is None:
                return False
            self._save()
            return True

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        payload = _RECORDS.dump_json(list(self._records.values()), indent=2)
        tmp.write_bytes(payload)
        os.replace(tmp, self._path)
