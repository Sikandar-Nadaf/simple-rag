from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.api.models.Collection import Collection

from rag_app.models import RetrievedChunk, SourceInfo


@dataclass(slots=True)
class StoredChunk:
    chunk_id: str
    source: str
    chunk_index: int
    text: str
    embedding: list[float]


class VectorStore(Protocol):
    def add_chunks(self, chunks: list[StoredChunk], rebuild: bool = False) -> int:
        ...

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        ...

    def list_sources(self) -> list[SourceInfo]:
        ...

    def count_chunks(self) -> int:
        ...


class ChromaVectorStore:
    def __init__(self, persist_directory: Path, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(name=collection_name)

    def _reset_collection(self) -> Collection:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(name=self._collection_name)
        return self._collection

    def add_chunks(self, chunks: list[StoredChunk], rebuild: bool = False) -> int:
        collection = self._reset_collection() if rebuild else self._collection
        if not chunks:
            return 0
        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            metadatas=[
                {"source": chunk.source, "chunk_index": chunk.chunk_index}
                for chunk in chunks
            ],
        )
        return len(chunks)

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        chunks: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            chunks.append(
                RetrievedChunk(
                    source=str(metadata["source"]),
                    chunk_index=int(metadata["chunk_index"]),
                    text=str(document),
                    score=float(distance),
                )
            )
        return chunks

    def list_sources(self) -> list[SourceInfo]:
        payload = self._collection.get(include=["metadatas"])
        metadatas = payload.get("metadatas") or []
        counts = Counter(str(metadata["source"]) for metadata in metadatas)
        return [
            SourceInfo(source=source, chunk_count=count)
            for source, count in sorted(counts.items())
        ]

    def count_chunks(self) -> int:
        return int(self._collection.count())


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, StoredChunk] = {}

    def add_chunks(self, chunks: list[StoredChunk], rebuild: bool = False) -> int:
        if rebuild:
            self._chunks.clear()
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
        return len(chunks)

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        ranked_chunks = sorted(
            self._chunks.values(),
            key=lambda chunk: _euclidean_distance(query_embedding, chunk.embedding),
        )
        return [
            RetrievedChunk(
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                score=_euclidean_distance(query_embedding, chunk.embedding),
            )
            for chunk in ranked_chunks[:top_k]
        ]

    def list_sources(self) -> list[SourceInfo]:
        counts = Counter(chunk.source for chunk in self._chunks.values())
        return [
            SourceInfo(source=source, chunk_count=count)
            for source, count in sorted(counts.items())
        ]

    def count_chunks(self) -> int:
        return len(self._chunks)


def create_vector_store(
    backend: str,
    persist_directory: Path,
    collection_name: str,
) -> VectorStore:
    normalized_backend = backend.strip().lower()
    if normalized_backend == "chroma":
        return ChromaVectorStore(
            persist_directory=persist_directory,
            collection_name=collection_name,
        )
    if normalized_backend == "memory":
        return InMemoryVectorStore()
    raise ValueError(
        f"Unsupported vector backend '{backend}'. Supported backends: chroma, memory."
    )


def _euclidean_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions must match for similarity search")
    return sum((lhs - rhs) ** 2 for lhs, rhs in zip(left, right)) ** 0.5
