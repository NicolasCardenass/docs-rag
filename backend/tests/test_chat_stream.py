import json
from collections.abc import Iterator, Sequence
from typing import Any

from fastapi.testclient import TestClient

from docs_rag.chat import NO_CONTEXT_ANSWER, ChatService
from docs_rag.llm import LLMError, Message

SOLAR = "Solar panels convert sunlight into electricity using photovoltaic cells."


def read_events(client: TestClient, payload: dict[str, Any]) -> list[tuple[str, Any]]:
    with client.stream("POST", "/chat/stream", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = response.read().decode()

    events = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def test_streams_sources_tokens_and_done(client: TestClient) -> None:
    client.post("/documents", files={"file": ("solar.md", SOLAR.encode())})

    events = read_events(client, {"question": "How do solar panels work?"})

    names = [name for name, _ in events]
    assert names[0] == "sources"
    assert names[-1] == "done"
    assert set(names[1:-1]) == {"token"}

    sources = events[0][1]
    assert sources[0]["number"] == 1
    assert sources[0]["filename"] == "solar.md"

    done = events[-1][1]
    tokens = "".join(data["text"] for name, data in events if name == "token")
    assert done["answer"] == tokens
    assert done["cited"] == [1]
    assert done["model"] == "mock:extractive"


def test_streams_fallback_without_documents(client: TestClient) -> None:
    events = read_events(client, {"question": "Anything?"})

    assert events[0] == ("sources", [])
    assert events[-1][1]["answer"] == NO_CONTEXT_ANSWER
    assert events[-1][1]["cited"] == []


def test_reports_errors_raised_during_generation(client: TestClient) -> None:
    class FailingModel:
        name = "fake:failing"

        def generate(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
            raise NotImplementedError

        def stream(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> Iterator[str]:
            yield "Partial"
            raise LLMError("connection lost")

    client.post("/documents", files={"file": ("solar.md", SOLAR.encode())})
    retriever = client.app.state.retriever  # type: ignore[attr-defined]
    client.app.state.chat_service = ChatService(retriever, FailingModel())  # type: ignore[attr-defined]

    events = read_events(client, {"question": "solar panels"})

    assert [name for name, _ in events] == ["sources", "token", "error"]
    assert events[-1][1] == {"detail": "connection lost"}


def test_validates_request(client: TestClient) -> None:
    assert client.post("/chat/stream", json={"question": ""}).status_code == 422
