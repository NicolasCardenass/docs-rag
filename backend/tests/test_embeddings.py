import json
import math

import httpx2
import openai
import pytest

from docs_rag.config import Settings
from docs_rag.embeddings import (
    EmbeddingError,
    MockEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    create_embedder,
)


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class TestMockEmbedder:
    def test_vectors_are_deterministic_and_normalized(self) -> None:
        embedder = MockEmbedder(dimension=64)

        first, second = embedder.embed_documents(["Vector stores", "Vector stores"])

        assert first == second
        assert len(first) == 64
        assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0)

    def test_related_texts_are_more_similar(self) -> None:
        embedder = MockEmbedder()
        query = embedder.embed_query("How are documents split into chunks?")
        related, unrelated = embedder.embed_documents(
            ["Documents are split into chunks with overlap.", "The weather is sunny today."]
        )

        assert cosine(query, related) > cosine(query, unrelated)

    def test_empty_text_gives_zero_vector(self) -> None:
        assert set(MockEmbedder(dimension=8).embed_query("")) == {0.0}

    def test_rejects_invalid_dimension(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            MockEmbedder(dimension=0)


def openai_client(handler: httpx2.MockTransport) -> openai.OpenAI:
    return openai.OpenAI(
        api_key="test-key",
        base_url="https://api.test/v1",
        http_client=httpx2.Client(transport=handler),
        max_retries=0,
    )


class TestOpenAIEmbedder:
    def test_batches_requests_and_keeps_order(self) -> None:
        batches: list[list[str]] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content)
            batches.append(body["input"])
            # Return items in reverse order to check that results are re-sorted.
            data = [
                {"object": "embedding", "index": i, "embedding": [float(len(text))]}
                for i, text in reversed(list(enumerate(body["input"])))
            ]
            return httpx2.Response(
                200,
                json={
                    "object": "list",
                    "data": data,
                    "model": body["model"],
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
                },
            )

        embedder = OpenAIEmbedder(openai_client(httpx2.MockTransport(handler)), "m", batch_size=2)

        vectors = embedder.embed_documents(["a", "bb", "ccc"])

        assert batches == [["a", "bb"], ["ccc"]]
        assert vectors == [[1.0], [2.0], [3.0]]
        assert embedder.name == "openai:m"

    def test_wraps_api_errors(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(401, json={"error": {"message": "bad key"}})

        embedder = OpenAIEmbedder(openai_client(httpx2.MockTransport(handler)), "m")

        with pytest.raises(EmbeddingError, match="OpenAI embeddings request failed"):
            embedder.embed_query("hello")


class TestOllamaEmbedder:
    def test_calls_embed_endpoint(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            assert request.url.path == "/api/embed"
            body = json.loads(request.content)
            assert body == {"model": "nomic-embed-text", "input": ["one", "two"]}
            return httpx2.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

        client = httpx2.Client(base_url="http://ollama", transport=httpx2.MockTransport(handler))
        embedder = OllamaEmbedder(client, "nomic-embed-text")

        assert embedder.embed_documents(["one", "two"]) == [[0.1, 0.2], [0.3, 0.4]]

    def test_rejects_unexpected_payload_and_http_errors(self) -> None:
        responses = iter(
            [httpx2.Response(200, json={"embeddings": []}), httpx2.Response(500, text="boom")]
        )
        client = httpx2.Client(
            base_url="http://ollama", transport=httpx2.MockTransport(lambda _: next(responses))
        )
        embedder = OllamaEmbedder(client, "m")

        with pytest.raises(EmbeddingError, match="unexpected"):
            embedder.embed_query("x")
        with pytest.raises(EmbeddingError, match="request failed"):
            embedder.embed_query("x")

    def test_empty_input_makes_no_request(self) -> None:
        client = httpx2.Client(transport=httpx2.MockTransport(lambda _: pytest.fail("no call")))

        assert OllamaEmbedder(client, "m").embed_documents([]) == []


@pytest.mark.parametrize(
    ("overrides", "expected_prefix"),
    [
        ({}, "mock:"),
        ({"embedding_provider": "ollama"}, "ollama:"),
        ({"embedding_provider": "openai", "openai_api_key": "sk-test"}, "openai:"),
    ],
)
def test_factory_selects_provider(overrides: dict[str, str], expected_prefix: str) -> None:
    settings = Settings(_env_file=None, **overrides)  # type: ignore[call-arg, arg-type]

    assert create_embedder(settings).name.startswith(expected_prefix)
