"""Messages from the Anthropic API."""

from collections.abc import Iterator, Sequence
from typing import Any

import anthropic

from docs_rag.llm.base import LLMError, Message, split_system


class AnthropicChatModel:
    def __init__(self, client: anthropic.Anthropic, model: str) -> None:
        self._client = client
        self._model = model

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def _params(self, messages: Sequence[Message], max_tokens: int) -> dict[str, Any]:
        system, conversation = split_system(messages)
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in conversation],
        }
        if system:
            params["system"] = system
        return params

    def generate(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
        try:
            response = self._client.messages.create(**self._params(messages, max_tokens))
        except anthropic.AnthropicError as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
        return "".join(block.text for block in response.content if block.type == "text")

    def stream(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> Iterator[str]:
        try:
            events = self._client.messages.create(**self._params(messages, max_tokens), stream=True)
            for event in events:
                if event.type == "content_block_delta" and event.delta.type == "text_delta":
                    yield event.delta.text
        except anthropic.AnthropicError as exc:
            raise LLMError(f"Anthropic request failed: {exc}") from exc
