import itertools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from docs_rag.ingestion import DocumentText, PageText, chunk_document, split_text

PARAGRAPH = (
    "Retrieval-augmented generation combines search with a language model. "
    "Relevant passages are retrieved first. The model then answers using them."
)


def test_short_text_is_a_single_chunk() -> None:
    assert split_text("Hello world.", chunk_size=100, overlap=10) == [(0, "Hello world.")]


def test_empty_or_blank_text_has_no_chunks() -> None:
    assert split_text("", chunk_size=100, overlap=10) == []
    assert split_text("   \n\n ", chunk_size=100, overlap=10) == []


def test_chunks_respect_size_and_prefer_sentence_boundaries() -> None:
    text = " ".join(f"Sentence number {i} is short." for i in range(20))

    chunks = split_text(text, chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for _, chunk in chunks)
    # Sentences are short, so a sentence end always fits in the second half
    # of the window: every chunk except the last ends at the end of a sentence.
    assert all(chunk.endswith(".") for _, chunk in chunks[:-1])


def test_prefers_paragraph_breaks() -> None:
    text = "First paragraph about embeddings.\n\nSecond paragraph about vector stores."

    chunks = split_text(text, chunk_size=60, overlap=0)

    assert [chunk for _, chunk in chunks] == [
        "First paragraph about embeddings.",
        "Second paragraph about vector stores.",
    ]


def test_consecutive_chunks_overlap() -> None:
    words = " ".join(f"word{i}" for i in range(200))

    chunks = split_text(words, chunk_size=100, overlap=30)

    for (_, previous), (_, current) in itertools.pairwise(chunks):
        first_word = current.split()[0]
        assert first_word in previous, "next chunk should start inside the previous one"


def test_offsets_point_to_chunk_text() -> None:
    text = "  Leading spaces. " + PARAGRAPH * 2

    for start, chunk in split_text(text, chunk_size=80, overlap=15):
        assert text[start : start + len(chunk)] == chunk


def test_hard_cut_when_there_is_no_boundary() -> None:
    chunks = split_text("x" * 250, chunk_size=100, overlap=0)

    assert [len(chunk) for _, chunk in chunks] == [100, 100, 50]


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (-5, 0), (100, 100), (100, -1)])
def test_rejects_invalid_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="must be"):
        split_text("text", chunk_size=size, overlap=overlap)


def test_chunk_document_keeps_page_numbers_and_global_indexes() -> None:
    document = DocumentText(
        filename="doc.pdf",
        media_type="application/pdf",
        pages=(PageText(1, PARAGRAPH * 2), PageText(3, "Short last page.")),
    )

    chunks = chunk_document(document, chunk_size=100, overlap=10)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert {chunk.page for chunk in chunks} == {1, 3}
    assert chunks[-1].page == 3
    assert chunks[-1].text == "Short last page."


@settings(max_examples=200)
@given(
    text=st.text(alphabet=st.sampled_from("abc .\n"), max_size=600),
    size=st.integers(min_value=10, max_value=200),
    overlap_ratio=st.floats(min_value=0, max_value=0.9),
)
def test_properties_hold_for_arbitrary_text(text: str, size: int, overlap_ratio: float) -> None:
    overlap = int(size * overlap_ratio)

    chunks = split_text(text, chunk_size=size, overlap=overlap)

    # Size limit, faithful offsets and no lost content.
    assert all(0 < len(chunk) <= size for _, chunk in chunks)
    assert all(text[start : start + len(chunk)] == chunk for start, chunk in chunks)
    covered = set()
    for start, chunk in chunks:
        covered.update(range(start, start + len(chunk)))
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())
