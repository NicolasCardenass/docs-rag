"""Chat completions from the OpenAI API."""

from collections.abc import Iterator, Sequence
from typing import Any

import openai

from docs_rag.llm.base import LLMError, Message


class OpenAIChatModel:
    def __init__(self, client: openai.OpenAI, model: str, temperature: float = 0.2) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    @property
    def name(self) -> str:
        return f"openai:{self._model}"

    def _payload(self, messages: Sequence[Message]) -> list[Any]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def generate(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=self._payload(messages),
                max_tokens=max_tokens,
                temperature=self._temperature,
            )
        except openai.OpenAIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        return response.choices[0].message.content or ""

    def stream(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> Iterator[str]:
        try:
            chunks = self._client.chat.completions.create(
                model=self._model,
                messages=self._payload(messages),
                max_tokens=max_tokens,
                temperature=self._temperature,
                stream=True,
            )
            for chunk in chunks:
                if chunk.choices and (text := chunk.choices[0].delta.content):
                    yield text
        except openai.OpenAIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
