from __future__ import annotations

from collections.abc import Sequence

from rag_app.models import RetrievedChunk


SYSTEM_PROMPT = (
    "You answer questions using only the provided document context. "
    "If the context is insufficient, say that you do not have enough information from the indexed documents. "
    "Always cite the source names you used."
)


def build_context_block(chunks: Sequence[RetrievedChunk]) -> str:
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        lines.append(
            f"[{index}] source={chunk.source} chunk={chunk.chunk_index}\n{chunk.text}"
        )
    return "\n\n".join(lines)


def build_messages(question: str, chunks: Sequence[RetrievedChunk]) -> list[dict[str, str]]:
    context = build_context_block(chunks)
    user_prompt = (
        "Answer the question using only the context below.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "If the context does not answer the question, say so clearly."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
