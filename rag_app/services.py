from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_app.chunking import chunk_text
from rag_app.config import Settings
from rag_app.documents import load_documents
from rag_app.models import ChatResponse, IngestResponse
from rag_app.prompting import build_messages
from rag_app.providers.base import ChatProvider, EmbeddingProvider
from rag_app.vectorstore import ChromaVectorStore, StoredChunk


@dataclass(slots=True)
class RagService:
    settings: Settings
    embedding_provider: EmbeddingProvider
    chat_provider: ChatProvider
    vector_store: ChromaVectorStore

    def ingest(self, rebuild: bool = False) -> IngestResponse:
        documents = load_documents(self.settings.data_dir)
        pending_chunks: list[tuple[str, int, str]] = []
        for document in documents:
            for chunk_index, text in enumerate(
                chunk_text(
                    document.text,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
            ):
                pending_chunks.append((document.source, chunk_index, text))

        embeddings = (
            self.embedding_provider.embed_texts([chunk[2] for chunk in pending_chunks])
            if pending_chunks
            else []
        )
        stored_chunks = [
            StoredChunk(
                chunk_id=f"{source}:{chunk_index}",
                source=source,
                chunk_index=chunk_index,
                text=text,
                embedding=embedding,
            )
            for (source, chunk_index, text), embedding in zip(pending_chunks, embeddings)
        ]
        ingested_chunks = self.vector_store.add_chunks(stored_chunks, rebuild=rebuild)
        return IngestResponse(
            ingested_files=len(documents),
            ingested_chunks=ingested_chunks,
            sources=self.vector_store.list_sources(),
        )

    def chat(self, question: str, top_k: int | None = None) -> ChatResponse:
        effective_top_k = top_k or self.settings.top_k
        if self.vector_store.count_chunks() == 0:
            return ChatResponse(
                answer="No documents have been indexed yet. Ingest documents first.",
                retrieved_chunks=[],
            )

        query_embedding = self.embedding_provider.embed_query(question)
        retrieved_chunks = self.vector_store.query(query_embedding, effective_top_k)
        if not retrieved_chunks:
            return ChatResponse(
                answer="I do not have enough information from the indexed documents to answer that.",
                retrieved_chunks=[],
            )

        messages = build_messages(question, retrieved_chunks)
        answer = self.chat_provider.chat(messages)
        return ChatResponse(answer=answer, retrieved_chunks=retrieved_chunks)

    def sources(self):
        return self.vector_store.list_sources()

    def save_upload(self, filename: str, content: bytes) -> Path:
        target = self.settings.data_dir / Path(filename).name
        target.write_bytes(content)
        return target
