"""Deterministic chat model for development and tests."""

import re
from collections.abc import Iterator, Sequence

from docs_rag.llm.base import Message

# Context passages are formatted as "[1] text" by the RAG prompt.
# A passage runs until the next "[n]" marker, a blank line or the end.
_SOURCE = re.compile(r"^\[(\d+)\]\s*(.+?)(?=^\[\d+\]|\n\s*\n|\Z)", re.MULTILINE | re.DOTALL)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


class MockChatModel:
    """Answers by quoting the first sentence of the first context passages.

    It needs no model or network, so the whole question-answering flow can
    be run and tested offline. It does not reason or summarize.
    """

    def __init__(self, max_sources: int = 2) -> None:
        self._max_sources = max_sources

    @property
    def name(self) -> str:
        return "mock:extractive"

    def generate(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
        prompt = next((m.content for m in reversed(messages) if m.role == "user"), "")
        sources = _SOURCE.findall(prompt)[: self._max_sources]
        if not sources:
            return "I could not find relevant information in the provided documents."
        sentences = [
            f"{_SENTENCE_END.split(text.strip(), maxsplit=1)[0].strip()} [{number}]"
            for number, text in sources
        ]
        answer = "According to the documents: " + " ".join(sentences)
        return " ".join(answer.split()[:max_tokens])

    def stream(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> Iterator[str]:
        words = self.generate(messages, max_tokens=max_tokens).split(" ")
        for position, word in enumerate(words):
            yield word if position == 0 else f" {word}"
