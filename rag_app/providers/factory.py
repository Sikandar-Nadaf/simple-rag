from __future__ import annotations

from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.providers.ollama import OllamaProvider
from rag_app.providers.simple import EchoChatProvider, SimpleEmbeddingProvider


def create_embedding_provider(
    backend: str,
    base_url: str,
    embedding_model: str,
    chat_model: str,
) -> EmbeddingProvider:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "ollama":
        return OllamaProvider(
            base_url=base_url,
            embedding_model=embedding_model,
            chat_model=chat_model,
        )
    if normalized_backend == "simple":
        return SimpleEmbeddingProvider()
    raise ValueError(
        f"Unsupported embedding backend '{backend}'. Supported backends: ollama, simple."
    )


def create_chat_provider(
    backend: str,
    base_url: str,
    embedding_model: str,
    chat_model: str,
) -> ChatProvider:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "ollama":
        return OllamaProvider(
            base_url=base_url,
            embedding_model=embedding_model,
            chat_model=chat_model,
        )
    if normalized_backend == "echo":
        return EchoChatProvider()
    raise ValueError(
        f"Unsupported chat backend '{backend}'. Supported backends: ollama, echo."
    )
