from collections.abc import Iterator, Sequence

from fastapi.testclient import TestClient

from docs_rag.chat import (
    NO_CONTEXT_ANSWER,
    ChatService,
    build_messages,
    cited_numbers,
)
from docs_rag.llm import LLMError, Message
from docs_rag.retrieval import Retriever
from docs_rag.store import SearchResult

SOLAR = (
    "Solar panels convert sunlight into electricity using photovoltaic cells.\n\n"
    "Inverters convert the direct current from the panels into alternating current."
)


def result(text: str, page: int = 1) -> SearchResult:
    return SearchResult("d:0", "d", "doc.pdf", page, 0, text, 0.9)


class RecordingModel:
    """Fake chat model that records the prompt and returns a fixed answer."""

    def __init__(self, reply: str = "Answer [1].") -> None:
        self.reply = reply
        self.calls: list[list[Message]] = []

    @property
    def name(self) -> str:
        return "fake:recording"

    def generate(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
        self.calls.append(list(messages))
        return self.reply

    def stream(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> Iterator[str]:
        yield self.generate(messages, max_tokens=max_tokens)


class FakeRetriever(Retriever):
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.default_top_k = 4

    def search(
        self, query: str, top_k: int | None = None, document_ids: Sequence[str] | None = None
    ) -> list[SearchResult]:
        return self._results


def test_prompt_numbers_passages_and_names_sources() -> None:
    messages = build_messages("What is it?", [result("Alpha."), result("Beta.", page=3)])

    assert messages[0].role == "system"
    assert "Cite every statement" in messages[0].content
    user = messages[1].content
    assert "[1] (source: doc.pdf, page 1)\nAlpha." in user
    assert "[2] (source: doc.pdf, page 3)\nBeta." in user
    assert user.endswith("Question: What is it?")


def test_cited_numbers_ignores_invalid_references() -> None:
    assert cited_numbers("See [1] and [3][2], not [9] or [0].", available=3) == {1, 2, 3}


def test_answer_marks_cited_sources() -> None:
    model = RecordingModel("Panels make power [2].")
    service = ChatService(FakeRetriever([result("A"), result("B")]), model)

    answer = service.answer("question")

    assert answer.answer == "Panels make power [2]."
    assert [(s.number, s.cited) for s in answer.sources] == [(1, False), (2, True)]
    assert answer.model == "fake:recording"


def test_does_not_call_llm_without_context() -> None:
    model = RecordingModel()
    service = ChatService(FakeRetriever([]), model)

    answer = service.answer("anything")

    assert answer.answer == NO_CONTEXT_ANSWER
    assert answer.sources == []
    assert model.calls == []


def test_chat_endpoint_answers_with_citations(client: TestClient) -> None:
    upload = client.post("/documents", files={"file": ("solar.md", SOLAR.encode())})
    document_id = upload.json()["id"]

    response = client.post("/chat", json={"question": "How do solar panels make electricity?"})

    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "mock:extractive"
    assert "[1]" in body["answer"]
    first = body["sources"][0]
    assert first["number"] == 1
    assert first["cited"] is True
    assert first["document_id"] == document_id
    assert first["filename"] == "solar.md"
    assert first["page"] == 1


def test_chat_endpoint_without_documents(client: TestClient) -> None:
    body = client.post("/chat", json={"question": "Anything?"}).json()

    assert body["answer"] == NO_CONTEXT_ANSWER
    assert body["sources"] == []


def test_chat_endpoint_validates_and_reports_provider_errors(client: TestClient) -> None:
    assert client.post("/chat", json={"question": ""}).status_code == 422

    class FailingModel(RecordingModel):
        def generate(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
            raise LLMError("provider down")

    client.app.state.chat_service = ChatService(  # type: ignore[attr-defined]
        FakeRetriever([result("A")]), FailingModel()
    )
    response = client.post("/chat", json={"question": "Q?"})

    assert response.status_code == 502
    assert response.json()["detail"] == "provider down"
