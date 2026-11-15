from pathlib import Path

import chromadb
import pytest

from docs_rag.embeddings import MockEmbedder
from docs_rag.ingestion import Chunk
from docs_rag.store import ChromaVectorStore, VectorStoreError

EMBEDDER = MockEmbedder()


def make_store(path: Path) -> ChromaVectorStore:
    return ChromaVectorStore.persistent(path, EMBEDDER.name)


def add_document(store: ChromaVectorStore, document_id: str, texts: list[str]) -> None:
    chunks = [Chunk(index=i, page=i + 1, start=0, text=text) for i, text in enumerate(texts)]
    store.add_chunks(document_id, f"{document_id}.pdf", chunks, EMBEDDER.embed_documents(texts))


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    store = make_store(tmp_path / "index")
    add_document(
        store,
        "solar",
        ["Solar panels convert sunlight into electricity.", "Inverters convert DC to AC."],
    )
    add_document(store, "cooking", ["Boil the pasta in salted water for ten minutes."])
    return store


def test_search_returns_most_similar_chunks_with_metadata(store: ChromaVectorStore) -> None:
    results = store.search(EMBEDDER.embed_query("How do solar panels make electricity?"), top_k=2)

    best = results[0]
    assert best.document_id == "solar"
    assert best.filename == "solar.pdf"
    assert best.page == 1
    assert best.chunk_id == "solar:0"
    assert "Solar panels" in best.text
    assert results[0].score >= results[1].score
    assert -1.0 <= best.score <= 1.0


def test_search_can_be_restricted_to_documents(store: ChromaVectorStore) -> None:
    query = EMBEDDER.embed_query("solar panels")

    results = store.search(query, top_k=5, document_ids=["cooking"])

    assert {result.document_id for result in results} == {"cooking"}
    assert store.search(query, top_k=5, document_ids=[]) == []


def test_top_k_limits_results(store: ChromaVectorStore) -> None:
    assert len(store.search(EMBEDDER.embed_query("convert"), top_k=1)) == 1
    assert store.search(EMBEDDER.embed_query("convert"), top_k=0) == []


def test_delete_document_removes_only_its_chunks(store: ChromaVectorStore) -> None:
    assert store.count() == 3

    assert store.delete_document("solar") == 2
    assert store.delete_document("solar") == 0
    assert store.count() == 1
    assert store.count("cooking") == 1


def test_data_persists_across_instances(tmp_path: Path) -> None:
    add_document(make_store(tmp_path / "index"), "doc", ["Persistent text."])

    assert make_store(tmp_path / "index").count("doc") == 1


def test_rejects_vectors_from_a_different_embedder(tmp_path: Path) -> None:
    make_store(tmp_path / "index")

    with pytest.raises(VectorStoreError, match="was built with embedder"):
        ChromaVectorStore.persistent(tmp_path / "index", "openai:text-embedding-3-small")


def test_search_on_empty_store_and_length_mismatch() -> None:
    store = ChromaVectorStore(chromadb.EphemeralClient(), EMBEDDER.name, "empty-test")

    assert store.search(EMBEDDER.embed_query("anything"), top_k=3) == []
    with pytest.raises(ValueError, match="same length"):
        store.add_chunks("d", "d.txt", [Chunk(0, 1, 0, "text")], [])
