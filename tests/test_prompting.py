from rag_app.models import RetrievedChunk
from rag_app.prompting import build_messages


def test_build_messages_includes_question_and_sources():
    chunks = [
        RetrievedChunk(source="notes.md", chunk_index=0, text="RAG is retrieval augmented generation.", score=0.1),
        RetrievedChunk(source="guide.txt", chunk_index=1, text="Chroma stores vectors locally.", score=0.2),
    ]

    messages = build_messages("What is RAG?", chunks)

    assert messages[0]["role"] == "system"
    assert "only the provided document context" in messages[0]["content"]
    assert "Question: What is RAG?" in messages[1]["content"]
    assert "source=notes.md" in messages[1]["content"]
    assert "source=guide.txt" in messages[1]["content"]
