from __future__ import annotations

from dataclasses import dataclass

from rag_app.providers.base import ChatProvider, EmbeddingProvider


@dataclass(slots=True)
class SimpleEmbeddingProvider(EmbeddingProvider):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return _embed_text(text)


@dataclass(slots=True)
class EchoChatProvider(ChatProvider):
    prefix: str = "Echo response based on provided context:"

    def chat(self, messages: list[dict[str, str]]) -> str:
        last_user_message = next(
            (message.get("content", "") for message in reversed(messages) if message.get("role") == "user"),
            "",
        )
        return f"{self.prefix} {last_user_message}".strip()


def _embed_text(text: str) -> list[float]:
    normalized = text.strip().lower()
    if not normalized:
        return [0.0, 0.0, 0.0, 0.0]

    vowels = sum(1 for char in normalized if char in "aeiou")
    whitespace = sum(1 for char in normalized if char.isspace())
    rag_mentions = float(normalized.count("rag"))
    ascii_signal = float(sum(ord(char) for char in normalized) % 997) / 997.0
    return [float(len(normalized)), float(vowels), float(whitespace), rag_mentions + ascii_signal]
