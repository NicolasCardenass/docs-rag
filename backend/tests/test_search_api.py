from fastapi.testclient import TestClient

SOLAR = (
    "Solar panels convert sunlight into electricity using photovoltaic cells.\n\n"
    "Inverters convert the direct current from the panels into alternating current."
)
PASTA = "Boil the pasta in salted water for ten minutes, then drain it."


def upload(client: TestClient, filename: str, text: str) -> str:
    response = client.post("/documents", files={"file": (filename, text.encode())})
    assert response.status_code == 201, response.text
    document_id: str = response.json()["id"]
    return document_id


def test_search_ranks_relevant_chunks_first(client: TestClient) -> None:
    solar_id = upload(client, "solar.md", SOLAR)
    upload(client, "pasta.txt", PASTA)

    response = client.post("/search", json={"query": "How do solar panels produce electricity?"})

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "How do solar panels produce electricity?"
    best = body["results"][0]
    assert best["document_id"] == solar_id
    assert best["filename"] == "solar.md"
    assert best["page"] == 1
    assert "Solar panels" in best["text"]
    scores = [hit["score"] for hit in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_respects_top_k_and_document_filter(client: TestClient) -> None:
    upload(client, "solar.md", SOLAR)
    pasta_id = upload(client, "pasta.txt", PASTA)

    one = client.post("/search", json={"query": "convert", "top_k": 1}).json()["results"]
    only_pasta = client.post(
        "/search", json={"query": "solar panels", "document_ids": [pasta_id]}
    ).json()["results"]

    assert len(one) == 1
    assert {hit["document_id"] for hit in only_pasta} == {pasta_id}


def test_search_with_no_documents_returns_empty_results(client: TestClient) -> None:
    response = client.post("/search", json={"query": "anything"})

    assert response.status_code == 200
    assert response.json()["results"] == []


def test_search_validates_input(client: TestClient) -> None:
    assert client.post("/search", json={"query": ""}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 0}).status_code == 422
    assert client.post("/search", json={"query": "x", "top_k": 50}).status_code == 422
    assert client.post("/search", json={}).status_code == 422


def test_whitespace_query_returns_no_results(client: TestClient) -> None:
    upload(client, "solar.md", SOLAR)

    assert client.post("/search", json={"query": "   "}).json()["results"] == []
