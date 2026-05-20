from __future__ import annotations

from dataclasses import dataclass

import pytest

from rag_app.config import Settings
from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.services import RagService
from rag_app.vectorstore import ChromaVectorStore


@dataclass(slots=True)
class FakeProvider(EmbeddingProvider, ChatProvider):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(text.lower().count("rag"))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def chat(self, messages: list[dict[str, str]]) -> str:
        return f"stubbed answer based on {len(messages)} messages"


@pytest.fixture
def settings(tmp_path):
    data_dir = tmp_path / "data"
    chroma_dir = tmp_path / "chroma"
    data_dir.mkdir()
    chroma_dir.mkdir()
    return Settings(
        data_dir=data_dir,
        chroma_dir=chroma_dir,
        collection_name="test_collection",
        ollama_base_url="http://localhost:11434",
        embedding_model="fake-embed",
        chat_model="fake-chat",
        chunk_size=40,
        chunk_overlap=10,
        top_k=3,
    )


@pytest.fixture
def fake_service(settings):
    provider = FakeProvider()
    store = ChromaVectorStore(settings.chroma_dir, settings.collection_name)
    return RagService(
        settings=settings,
        embedding_provider=provider,
        chat_provider=provider,
        vector_store=store,
    )
