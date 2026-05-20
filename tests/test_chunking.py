import pytest

from rag_app.chunking import ParagraphChunker, chunk_text, create_chunker


def test_chunk_text_returns_single_chunk_for_small_text():
    assert chunk_text("small text", chunk_size=20, chunk_overlap=5) == ["small text"]


def test_chunk_text_uses_overlap():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=3)
    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]


def test_chunk_text_rejects_invalid_overlap():
    try:
        chunk_text("abc", chunk_size=5, chunk_overlap=5)
    except ValueError as exc:
        assert "smaller" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_paragraph_chunker_preserves_paragraph_boundaries_when_possible():
    text = "First paragraph.\n\nSecond paragraph."

    chunks = ParagraphChunker(chunk_size=20, chunk_overlap=5).chunk(text)

    assert chunks == ["First paragraph.", "Second paragraph."]


def test_paragraph_chunker_falls_back_to_fixed_size_for_long_paragraph():
    text = "abcdefghijklmnopqrstuvwxyz"

    chunks = ParagraphChunker(chunk_size=10, chunk_overlap=3).chunk(text)

    assert chunks == ["abcdefghij", "hijklmnopq", "opqrstuvwx", "vwxyz"]


@pytest.mark.parametrize("strategy_name", ["fixed", "paragraph"])
def test_create_chunker_supports_known_strategies(strategy_name):
    chunker = create_chunker(strategy_name, chunk_size=20, chunk_overlap=5)

    assert chunker.chunk("small text") == ["small text"]


def test_create_chunker_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        create_chunker("unknown", chunk_size=20, chunk_overlap=5)
