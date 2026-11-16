"""Chat model providers and the factory that selects one from settings."""

import anthropic
import httpx2
import openai

from docs_rag.config import Settings
from docs_rag.llm.anthropic_provider import AnthropicChatModel
from docs_rag.llm.base import ChatModel, LLMError, Message, Role
from docs_rag.llm.mock_provider import MockChatModel
from docs_rag.llm.ollama_provider import OllamaChatModel
from docs_rag.llm.openai_provider import OpenAIChatModel

__all__ = [
    "AnthropicChatModel",
    "ChatModel",
    "LLMError",
    "Message",
    "MockChatModel",
    "OllamaChatModel",
    "OpenAIChatModel",
    "Role",
    "create_chat_model",
]


def create_chat_model(settings: Settings) -> ChatModel:
    """Build the chat model selected by ``DOCS_RAG_LLM_PROVIDER``."""
    match settings.llm_provider:
        case "mock":
            return MockChatModel()
        case "openai":
            assert settings.openai_api_key is not None  # guaranteed by Settings validation
            client = openai.OpenAI(api_key=settings.openai_api_key.get_secret_value())
            return OpenAIChatModel(client, settings.openai_chat_model)
        case "anthropic":
            assert settings.anthropic_api_key is not None  # guaranteed by Settings validation
            anthropic_client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
            return AnthropicChatModel(anthropic_client, settings.anthropic_model)
        case "ollama":
            http = httpx2.Client(base_url=settings.ollama_base_url, timeout=120.0)
            return OllamaChatModel(http, settings.ollama_chat_model)
