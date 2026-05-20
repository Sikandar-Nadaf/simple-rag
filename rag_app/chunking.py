from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class ChunkingStrategy(Protocol):
    def chunk(self, text: str) -> list[str]:
        ...


@dataclass(slots=True)
class FixedSizeChunker:
    chunk_size: int
    chunk_overlap: int

    def chunk(self, text: str) -> list[str]:
        _validate_chunk_settings(self.chunk_size, self.chunk_overlap)

        normalized = text.strip()
        if not normalized:
            return []
        if len(normalized) <= self.chunk_size:
            return [normalized]

        chunks: list[str] = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        while start < len(normalized):
            end = min(len(normalized), start + self.chunk_size)
            chunk = normalized[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(normalized):
                break
            start += step
        return chunks


@dataclass(slots=True)
class ParagraphChunker:
    chunk_size: int
    chunk_overlap: int

    def chunk(self, text: str) -> list[str]:
        _validate_chunk_settings(self.chunk_size, self.chunk_overlap)

        normalized = text.strip()
        if not normalized:
            return []

        paragraphs = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
        if not paragraphs:
            return []

        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = paragraph if not current else f"{current}\n\n{paragraph}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)
            if len(paragraph) <= self.chunk_size:
                current = paragraph
            else:
                chunks.extend(FixedSizeChunker(self.chunk_size, self.chunk_overlap).chunk(paragraph))
                current = ""

        if current:
            chunks.append(current)
        return chunks


def create_chunker(
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> ChunkingStrategy:
    normalized_strategy = strategy.strip().lower()
    if normalized_strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if normalized_strategy == "paragraph":
        return ParagraphChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    raise ValueError(
        f"Unsupported chunking strategy '{strategy}'. Supported strategies: fixed, paragraph."
    )


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap).chunk(text)


def _validate_chunk_settings(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
