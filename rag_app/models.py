from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    rebuild: bool = False


class UploadResponse(BaseModel):
    filename: str
    path: str


class SourceInfo(BaseModel):
    source: str
    chunk_count: int


class IngestResponse(BaseModel):
    ingested_files: int
    ingested_chunks: int
    sources: list[SourceInfo]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)


class RetrievedChunk(BaseModel):
    source: str
    chunk_index: int
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunk]


class HealthResponse(BaseModel):
    status: str
    indexed_documents: int
    indexed_chunks: int
    settings: dict[str, object]
