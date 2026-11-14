"""Split document text into overlapping chunks for embedding.

Chunks are sized in characters. Each chunk ends at the most natural
boundary available near the size limit (paragraph, line, sentence, then
word), and consecutive chunks share ``overlap`` characters so that a
sentence cut at a boundary still appears whole in at least one chunk.
"""

from dataclasses import dataclass

from docs_rag.ingestion.extract import DocumentText

# Ordered from the most to the least preferred place to cut.
_SEPARATORS = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")


@dataclass(frozen=True, slots=True)
class Chunk:
    index: int
    """Position of the chunk within the whole document (0-based)."""
    page: int
    """Page the chunk comes from (1-based)."""
    start: int
    """Offset of the first character within the page text."""
    text: str


def _validate(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")


def _find_break(text: str, start: int, end: int) -> int:
    """Return where to cut ``text[start:end]``, preferring natural boundaries.

    Only the second half of the window is searched, so chunks never become
    much smaller than requested just to end on a nicer boundary.
    """
    min_end = start + (end - start) // 2
    for separator in _SEPARATORS:
        position = text.rfind(separator, min_end, end)
        if position != -1:
            return position + len(separator)
    return end  # no boundary: hard cut (e.g. a very long word or URL)


def split_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, str]]:
    """Split ``text`` into ``(start_offset, chunk_text)`` pairs.

    Every chunk has at most ``chunk_size`` characters and starts no later
    than ``overlap`` characters before the previous chunk ended.
    """
    _validate(chunk_size, overlap)
    chunks: list[tuple[int, str]] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            end = _find_break(text, start, end)

        piece = text[start:end]
        stripped = piece.strip()
        if stripped:
            chunks.append((start + (len(piece) - len(piece.lstrip())), stripped))

        if end >= length:
            break

        next_start = max(end - overlap, start + 1)
        # Begin the overlap at a word boundary rather than mid-word.
        if overlap and (space := text.find(" ", next_start, end)) != -1:
            next_start = space + 1
        start = next_start

    return chunks


def chunk_document(document: DocumentText, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk every page of a document, keeping page numbers for citations.

    Chunks never span two pages, so each one can be attributed to a page.
    """
    chunks: list[Chunk] = []
    for page in document.pages:
        for start, text in split_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(index=len(chunks), page=page.number, start=start, text=text))
    return chunks
