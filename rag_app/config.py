from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


@dataclass(slots=True)
class Settings:
    data_dir: Path
    vector_backend: str
    vector_store_dir: Path
    chunk_strategy: str
    embedding_backend: str
    chat_backend: str
    collection_name: str
    ollama_base_url: str
    embedding_model: str
    chat_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.getenv("RAG_DATA_DIR", "data")),
            vector_backend=os.getenv("RAG_VECTOR_BACKEND", "chroma"),
            vector_store_dir=Path(
                os.getenv("RAG_VECTOR_STORE_DIR", os.getenv("RAG_CHROMA_DIR", ".chroma"))
            ),
            chunk_strategy=os.getenv("RAG_CHUNK_STRATEGY", "fixed"),
            embedding_backend=os.getenv("RAG_EMBEDDING_BACKEND", "ollama"),
            chat_backend=os.getenv("RAG_CHAT_BACKEND", "ollama"),
            collection_name=os.getenv("RAG_COLLECTION_NAME", "rag_documents"),
            ollama_base_url=os.getenv("RAG_OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", "nomic-embed-text"),
            chat_model=os.getenv("RAG_CHAT_MODEL", "llama3.1"),
            chunk_size=_int_env("RAG_CHUNK_SIZE", 900),
            chunk_overlap=_int_env("RAG_CHUNK_OVERLAP", 150),
            top_k=_int_env("RAG_TOP_K", 4),
        )

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict[str, object]:
        return {
            "data_dir": str(self.data_dir),
            "vector_backend": self.vector_backend,
            "vector_store_dir": str(self.vector_store_dir),
            "chunk_strategy": self.chunk_strategy,
            "embedding_backend": self.embedding_backend,
            "chat_backend": self.chat_backend,
            "collection_name": self.collection_name,
            "ollama_base_url": self.ollama_base_url,
            "embedding_model": self.embedding_model,
            "chat_model": self.chat_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
        }
