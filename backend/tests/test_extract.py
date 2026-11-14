import pytest
from fpdf import FPDF

from docs_rag.ingestion import (
    EmptyDocumentError,
    ExtractionError,
    UnsupportedFileTypeError,
    extract_text,
)
from docs_rag.ingestion.extract import normalize_whitespace


def make_pdf(*pages: str) -> bytes:
    pdf = FPDF()
    pdf.set_font("Helvetica", size=12)
    for text in pages:
        pdf.add_page()
        if text:
            pdf.multi_cell(0, 10, text)
    return bytes(pdf.output())


def test_extracts_pdf_text_per_page() -> None:
    document = extract_text("manual.pdf", make_pdf("First page text.", "Second page text."))

    assert document.media_type == "application/pdf"
    assert [page.number for page in document.pages] == [1, 2]
    assert "First page text." in document.pages[0].text
    assert "Second page text." in document.pages[1].text
    assert "First page" in document.text
    assert "Second page" in document.text


def test_skips_blank_pdf_pages_but_keeps_original_numbers() -> None:
    document = extract_text("report.pdf", make_pdf("Intro", "", "Conclusion"))

    assert [page.number for page in document.pages] == [1, 3]


def test_extracts_markdown_and_text_as_single_page() -> None:
    markdown = extract_text("README.MD", b"# Title\n\nSome   text.\n")
    plain = extract_text("notes.txt", b"Hello\r\nworld")

    assert markdown.media_type == "text/markdown"
    assert markdown.pages[0].number == 1
    assert markdown.text == "# Title\n\nSome text."
    assert plain.media_type == "text/plain"
    assert plain.text == "Hello\nworld"


def test_decodes_utf8_with_bom_and_latin1() -> None:
    assert extract_text("a.txt", "﻿ação".encode()).text == "ação"
    assert extract_text("b.txt", "ação".encode("latin-1")).text == "ação"


def test_rejects_unsupported_extension() -> None:
    with pytest.raises(UnsupportedFileTypeError, match=r"Unsupported file type '\.docx'"):
        extract_text("contract.docx", b"data")


def test_rejects_empty_documents() -> None:
    with pytest.raises(EmptyDocumentError):
        extract_text("empty.txt", b"   \n\n  ")
    with pytest.raises(EmptyDocumentError):
        extract_text("scanned.pdf", make_pdf(""))


def test_reports_corrupted_pdf() -> None:
    with pytest.raises(ExtractionError, match="Could not read PDF"):
        extract_text("broken.pdf", b"%PDF-1.7 this is not really a pdf")


def test_normalize_whitespace_keeps_paragraphs() -> None:
    raw = "Line  one \t\n\n\n\nLine\x00 two  \r\nend"
    assert normalize_whitespace(raw) == "Line one\n\nLine two\nend"
