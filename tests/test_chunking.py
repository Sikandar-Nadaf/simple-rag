from rag_app.chunking import chunk_text


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
