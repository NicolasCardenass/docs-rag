"""Extract plain text from uploaded files.

PDFs keep their page structure so answers can cite page numbers; plain text
and Markdown files are treated as a single page.
"""

import io
import re
from dataclasses import dataclass
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PdfReadError

SUPPORTED_EXTENSIONS = frozenset({".pdf", ".txt", ".md", ".markdown"})


class ExtractionError(Exception):
    """Base error for files whose text cannot be extracted."""


class UnsupportedFileTypeError(ExtractionError):
    def __init__(self, extension: str) -> None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        super().__init__(f"Unsupported file type '{extension or '(none)'}'. Supported: {supported}")
        self.extension = extension


class EmptyDocumentError(ExtractionError):
    def __init__(self, filename: str) -> None:
        super().__init__(f"No extractable text found in '{filename}'")


@dataclass(frozen=True, slots=True)
class PageText:
    number: int
    """1-based page number (always 1 for text files)."""
    text: str


@dataclass(frozen=True, slots=True)
class DocumentText:
    filename: str
    media_type: str
    pages: tuple[PageText, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    @property
    def char_count(self) -> int:
        return sum(len(page.text) for page in self.pages)


_HORIZONTAL_SPACE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces and blank lines while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = (_HORIZONTAL_SPACE.sub(" ", line).strip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        # Common in files exported by older Windows tools.
        return data.decode("latin-1")


def _extract_pdf(data: bytes, filename: str) -> list[PageText]:
    try:
        reader = PdfReader(io.BytesIO(data))
        raw_pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, ValueError) as exc:
        raise ExtractionError(f"Could not read PDF '{filename}': {exc}") from exc
    return [
        PageText(number=number, text=normalize_whitespace(raw))
        for number, raw in enumerate(raw_pages, start=1)
    ]


def extract_text(filename: str, data: bytes) -> DocumentText:
    """Extract normalized text from a PDF, plain-text or Markdown file.

    Raises:
        UnsupportedFileTypeError: the extension is not supported.
        EmptyDocumentError: the file contains no extractable text
            (e.g. a scanned PDF without a text layer).
        ExtractionError: the file is corrupted or cannot be parsed.
    """
    extension = PurePath(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(extension)

    if extension == ".pdf":
        media_type = "application/pdf"
        pages = _extract_pdf(data, filename)
    else:
        media_type = "text/markdown" if extension in {".md", ".markdown"} else "text/plain"
        pages = [PageText(number=1, text=normalize_whitespace(_decode(data)))]

    non_empty = tuple(page for page in pages if page.text)
    if not non_empty:
        raise EmptyDocumentError(filename)

    return DocumentText(filename=filename, media_type=media_type, pages=non_empty)
