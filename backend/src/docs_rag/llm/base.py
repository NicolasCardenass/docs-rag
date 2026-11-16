"""Chat model (LLM) provider interface."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class Message:
    role: Role
    content: str


class LLMError(Exception):
    """Raised when a provider fails to generate a response."""


class ChatModel(Protocol):
    """Generates assistant replies from a conversation."""

    @property
    def name(self) -> str:
        """Provider and model, e.g. ``"anthropic:claude-haiku-4-5"``."""
        ...

    def generate(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> str:
        """Return the full reply."""
        ...

    def stream(self, messages: Sequence[Message], *, max_tokens: int = 1024) -> Iterator[str]:
        """Yield the reply incrementally, as text fragments."""
        ...


def split_system(messages: Sequence[Message]) -> tuple[str | None, list[Message]]:
    """Separate system instructions from the conversation.

    Some APIs (e.g. Anthropic) take the system prompt as its own parameter.
    """
    system = "\n\n".join(m.content for m in messages if m.role == "system") or None
    return system, [m for m in messages if m.role != "system"]
