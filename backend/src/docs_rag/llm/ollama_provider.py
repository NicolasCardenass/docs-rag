"""Chat with a model served by a local Ollama server."""

import json
from collections.abc import Iterator, Sequence
from typing import Any

import httpx2

from docs_rag.llm.base import LLMError, Message


class OllamaChatModel:
    """Uses ``POST /api/chat``; streaming responses are newline-delimited JSON."""

    def __init__(self, client: httpx2.Client, model: str, temperature: float = 0.2) -> None:
        self._client = client
        self._model = model
        self._temperature = temperature

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def _body(self, messages: Sequence[Message], max_tokens: int, stream: bool) -> dict[str, Any]:
        return {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": stream,
            "options": {"temperature": self._temperature, "num_predict": max_tokens},
        }

    def generate(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
        try:
            response = self._client.post("/api/chat", json=self._body(messages, max_tokens, False))
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except (httpx2.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        return str(content)

    def stream(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> Iterator[str]:
        body = self._body(messages, max_tokens, True)
        try:
            with self._client.stream("POST", "/api/chat", json=body) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if text := event.get("message", {}).get("content"):
                        yield text
                    if event.get("done"):
                        break
        except (httpx2.HTTPError, ValueError) as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
