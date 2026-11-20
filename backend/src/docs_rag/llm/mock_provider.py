"""Deterministic chat model for development and tests."""

import re
from collections.abc import Iterator, Sequence

from docs_rag.llm.base import Message

# Context passages are formatted as "[n] (source: file, page p)" followed by
# the passage text. A passage runs until the next "[n]" marker, the question
# or the end of the prompt.
_SOURCE = re.compile(
    r"^\[(\d+)\][ \t]*(?:\(source:[^\n]*\)[ \t]*\n)?(.+?)(?=^\[\d+\]|\n\s*\nQuestion:|\Z)",
    re.MULTILINE | re.DOTALL,
)
_QUESTION = re.compile(r"^Question:\s*(.+)$", re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    ]
)

NO_ANSWER = "I could not find relevant information in the provided documents."


def _keywords(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _sentences(passage: str) -> list[str]:
    """Split a passage into sentences, ignoring Markdown headings."""
    lines = [line for line in passage.splitlines() if not line.lstrip().startswith("#")]
    text = " ".join(" ".join(lines).split())
    return [s for s in _SENTENCE_END.split(text) if s]


class MockChatModel:
    """Extractive model: quotes the sentences that best match the question.

    For each context passage it picks the sentence sharing the most words
    with the question and cites it. It needs no model or network, so the
    whole question-answering flow can be run and tested offline, but it does
    not reason, paraphrase or combine information.
    """

    def __init__(self, max_sources: int = 2) -> None:
        self._max_sources = max_sources

    @property
    def name(self) -> str:
        return "mock:extractive"

    def generate(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
        prompt = next((m.content for m in reversed(messages) if m.role == "user"), "")
        question = _QUESTION.search(prompt)
        question_words = _keywords(question.group(1)) if question else set()

        candidates: list[tuple[int, int, str]] = []  # (score, source number, sentence)
        for number, passage in _SOURCE.findall(prompt):
            sentences = _sentences(passage)
            if not sentences:
                continue
            best = max(sentences, key=lambda s: len(question_words & _keywords(s)))
            candidates.append((len(question_words & _keywords(best)), int(number), best))
        if not candidates:
            return NO_ANSWER

        # Keep the passages that match the question best (always at least one).
        candidates.sort(key=lambda c: (-c[0], c[1]))
        top_score = candidates[0][0]
        chosen = [c for c in candidates if c[0] == top_score or c[0] > 0][: self._max_sources]
        chosen.sort(key=lambda c: c[1])  # cite in passage order
        answer = " ".join(f"{sentence} [{number}]" for _, number, sentence in chosen)
        return " ".join(answer.split()[:max_tokens])

    def stream(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> Iterator[str]:
        words = self.generate(messages, max_tokens=max_tokens).split(" ")
        for position, word in enumerate(words):
            yield word if position == 0 else f" {word}"
