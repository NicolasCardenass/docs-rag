"""Retrieval-augmented question answering."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from docs_rag.llm import ChatModel, Message
from docs_rag.retrieval import Retriever
from docs_rag.store import SearchResult

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions about the user's documents.

Rules:
- Use ONLY the numbered context passages provided in the user message.
- Cite every statement with the number of its passage in square brackets, e.g. [1] or [2][3].
- If the passages do not contain the answer, say that you could not find it in the documents.
  Do not use outside knowledge and do not invent citations.
- Answer in the same language as the question, concisely.
"""

NO_CONTEXT_ANSWER = "I could not find relevant information in the indexed documents."

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True, slots=True)
class Source:
    number: int
    """The number used to cite this passage in the answer, e.g. 1 for [1]."""
    result: SearchResult
    cited: bool


@dataclass(frozen=True, slots=True)
class ChatAnswer:
    answer: str
    sources: list[Source]
    model: str


def format_context(results: Sequence[SearchResult]) -> str:
    """Number the passages so the model can cite them."""
    return "\n\n".join(
        f"[{number}] (source: {result.filename}, page {result.page})\n{result.text}"
        for number, result in enumerate(results, start=1)
    )


def build_messages(question: str, results: Sequence[SearchResult]) -> list[Message]:
    user = f"Context:\n{format_context(results)}\n\nQuestion: {question}"
    return [Message("system", SYSTEM_PROMPT), Message("user", user)]


def cited_numbers(answer: str, available: int) -> set[int]:
    """Citation numbers used in the answer that point to a real passage."""
    return {n for n in map(int, _CITATION.findall(answer)) if 1 <= n <= available}


class ChatService:
    """Answers questions by retrieving relevant chunks and asking the LLM."""

    def __init__(self, retriever: Retriever, model: ChatModel, max_tokens: int = 1024) -> None:
        self._retriever = retriever
        self._model = model
        self._max_tokens = max_tokens

    @property
    def model_name(self) -> str:
        return self._model.name

    def retrieve(
        self, question: str, top_k: int | None, document_ids: Sequence[str] | None
    ) -> list[SearchResult]:
        return self._retriever.search(question, top_k, document_ids)

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        document_ids: Sequence[str] | None = None,
    ) -> ChatAnswer:
        """Answer ``question`` from the documents, citing the passages used.

        When nothing relevant is retrieved the LLM is not called at all, which
        avoids answers made up without context.
        """
        results = self.retrieve(question, top_k, document_ids)
        if not results:
            return ChatAnswer(NO_CONTEXT_ANSWER, [], self.model_name)

        text = self._model.generate(build_messages(question, results), max_tokens=self._max_tokens)
        return ChatAnswer(text, make_sources(results, text), self.model_name)


def make_sources(results: Sequence[SearchResult], answer: str) -> list[Source]:
    cited = cited_numbers(answer, len(results))
    return [
        Source(number=number, result=result, cited=number in cited)
        for number, result in enumerate(results, start=1)
    ]
