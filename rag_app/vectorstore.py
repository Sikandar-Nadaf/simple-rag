from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

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
