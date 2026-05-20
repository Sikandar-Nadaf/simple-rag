from __future__ import annotations

from dataclasses import dataclass

import httpx

from rag_app.providers.base import ChatProvider, EmbeddingProvider


@dataclass(slots=True)
class OllamaProvider(EmbeddingProvider, ChatProvider):
    base_url: str
    embedding_model: str
    chat_model: str
    timeout: float = 60.0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.embedding_model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list):
            raise ValueError("Ollama embed response did not include embeddings")
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def chat(self, messages: list[dict[str, str]]) -> str:
        response = httpx.post(
            f"{self.base_url}/api/chat",
            json={"model": self.chat_model, "messages": messages, "stream": False},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        content = payload.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama chat response did not include message content")
        return content.strip()
