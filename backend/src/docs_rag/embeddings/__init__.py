"""Embedding providers and the factory that selects one from settings."""

import httpx2
import openai

from docs_rag.config import Settings
from docs_rag.embeddings.base import Embedder, EmbeddingError, Vector
from docs_rag.embeddings.mock_provider import MockEmbedder
from docs_rag.embeddings.ollama_provider import OllamaEmbedder
from docs_rag.embeddings.openai_provider import OpenAIEmbedder

__all__ = [
    "Embedder",
    "EmbeddingError",
    "MockEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "Vector",
    "create_embedder",
]


def create_embedder(settings: Settings) -> Embedder:
    """Build the embedding provider selected by ``DOCS_RAG_EMBEDDING_PROVIDER``."""
    match settings.embedding_provider:
        case "mock":
            return MockEmbedder()
        case "openai":
            assert settings.openai_api_key is not None  # guaranteed by Settings validation
            client = openai.OpenAI(api_key=settings.openai_api_key.get_secret_value())
            return OpenAIEmbedder(client, settings.openai_embedding_model)
        case "ollama":
            http = httpx2.Client(base_url=settings.ollama_base_url, timeout=60.0)
            return OllamaEmbedder(http, settings.ollama_embedding_model)
