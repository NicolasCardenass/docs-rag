import json
from collections.abc import Callable

import anthropic
import httpx2
import openai
import pytest

from docs_rag.config import Settings
from docs_rag.llm import (
    AnthropicChatModel,
    LLMError,
    Message,
    MockChatModel,
    OllamaChatModel,
    OpenAIChatModel,
    create_chat_model,
)
from docs_rag.llm.base import split_system

MESSAGES = [
    Message("system", "Answer using only the context."),
    Message(
        "user",
        "Context:\n[1] Solar panels make electricity. They use cells.\n"
        "[2] Inverters convert DC to AC.\n\nQuestion: How?",
    ),
]

Handler = Callable[[httpx2.Request], httpx2.Response]


def sse(*events: dict[str, object], done: bool = False) -> bytes:
    lines = [f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events]
    if done:
        lines.append("data: [DONE]\n\n")
    return "".join(lines).encode()


def test_split_system_separates_instructions() -> None:
    system, conversation = split_system(MESSAGES)

    assert system == "Answer using only the context."
    assert [m.role for m in conversation] == ["user"]


class TestMockChatModel:
    def test_quotes_first_sentence_of_sources_with_citations(self) -> None:
        answer = MockChatModel().generate(MESSAGES)

        assert answer == (
            "According to the documents: Solar panels make electricity. [1] "
            "Inverters convert DC to AC. [2]"
        )

    def test_stream_reassembles_into_full_answer(self) -> None:
        model = MockChatModel()

        assert "".join(model.stream(MESSAGES)) == model.generate(MESSAGES)

    def test_answers_without_sources(self) -> None:
        answer = MockChatModel().generate([Message("user", "Hello?")])

        assert "could not find" in answer


def openai_model(handler: Handler) -> OpenAIChatModel:
    client = openai.OpenAI(
        api_key="test",
        base_url="https://api.test/v1",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return OpenAIChatModel(client, "gpt-test")


def openai_chunk(text: str) -> dict[str, object]:
    return {
        "type": "chunk",
        "id": "c1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "gpt-test",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
    }


class TestOpenAIChatModel:
    def test_generate_sends_messages_and_returns_content(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content)
            assert body["model"] == "gpt-test"
            assert [m["role"] for m in body["messages"]] == ["system", "user"]
            return httpx2.Response(
                200,
                json={
                    "id": "x",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "gpt-test",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "Hi [1]"},
                        }
                    ],
                },
            )

        assert openai_model(handler).generate(MESSAGES) == "Hi [1]"

    def test_stream_yields_deltas(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            assert json.loads(request.content)["stream"] is True
            body = sse(openai_chunk("Hel"), openai_chunk("lo"), done=True)
            return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})

        assert list(openai_model(handler).stream(MESSAGES)) == ["Hel", "lo"]

    def test_wraps_errors(self) -> None:
        model = openai_model(lambda _: httpx2.Response(500, json={"error": {"message": "x"}}))

        with pytest.raises(LLMError, match="OpenAI request failed"):
            model.generate(MESSAGES)


def anthropic_model(handler: Handler) -> AnthropicChatModel:
    client = anthropic.Anthropic(
        api_key="test",
        base_url="https://api.test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return AnthropicChatModel(client, "model-test")


class TestAnthropicChatModel:
    def test_generate_sends_system_separately(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content)
            assert body["system"] == "Answer using only the context."
            assert [m["role"] for m in body["messages"]] == ["user"]
            return httpx2.Response(
                200,
                json={
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "model-test",
                    "content": [{"type": "text", "text": "Answer [1]"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                },
            )

        assert anthropic_model(handler).generate(MESSAGES) == "Answer [1]"

    def test_stream_yields_text_deltas(self) -> None:
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "model-test",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "An"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "swer"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ]

        def handler(request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(
                200, content=sse(*events), headers={"content-type": "text/event-stream"}
            )

        assert list(anthropic_model(handler).stream(MESSAGES)) == ["An", "swer"]

    def test_wraps_errors(self) -> None:
        model = anthropic_model(
            lambda _: httpx2.Response(
                401,
                json={"type": "error", "error": {"type": "authentication_error", "message": "x"}},
            )
        )

        with pytest.raises(LLMError, match="Anthropic request failed"):
            model.generate(MESSAGES)


def ollama_model(handler: Handler) -> OllamaChatModel:
    client = httpx2.Client(base_url="http://ollama", transport=httpx2.MockTransport(handler))
    return OllamaChatModel(client, "llama-test")


class TestOllamaChatModel:
    def test_generate(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            body = json.loads(request.content)
            assert request.url.path == "/api/chat"
            assert body["stream"] is False
            assert body["options"]["num_predict"] == 50
            return httpx2.Response(200, json={"message": {"role": "assistant", "content": "Ok"}})

        assert ollama_model(handler).generate(MESSAGES, max_tokens=50) == "Ok"

    def test_stream_reads_ndjson(self) -> None:
        lines = [
            {"message": {"content": "Hel"}, "done": False},
            {"message": {"content": "lo"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
        body = "\n".join(json.dumps(line) for line in lines).encode()

        model = ollama_model(lambda _: httpx2.Response(200, content=body))

        assert list(model.stream(MESSAGES)) == ["Hel", "lo"]

    def test_wraps_errors(self) -> None:
        model = ollama_model(lambda _: httpx2.Response(404, text="model not found"))

        with pytest.raises(LLMError, match="Ollama request failed"):
            model.generate(MESSAGES)
        with pytest.raises(LLMError, match="Ollama request failed"):
            list(model.stream(MESSAGES))


@pytest.mark.parametrize(
    ("overrides", "expected_prefix"),
    [
        ({}, "mock:"),
        ({"llm_provider": "ollama"}, "ollama:"),
        ({"llm_provider": "openai", "openai_api_key": "sk-test"}, "openai:"),
        ({"llm_provider": "anthropic", "anthropic_api_key": "sk-ant-test"}, "anthropic:"),
    ],
)
def test_factory_selects_provider(overrides: dict[str, str], expected_prefix: str) -> None:
    settings = Settings(_env_file=None, **overrides)  # type: ignore[call-arg, arg-type]

    assert create_chat_model(settings).name.startswith(expected_prefix)
