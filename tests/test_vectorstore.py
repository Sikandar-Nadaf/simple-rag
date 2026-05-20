from rag_app.vectorstore import ChromaVectorStore, StoredChunk


def test_vectorstore_adds_and_lists_sources(tmp_path):
    store = ChromaVectorStore(tmp_path / "chroma", "collection")
    added = store.add_chunks(
        [
            StoredChunk("a:0", "a.md", 0, "alpha rag", [1.0, 2.0]),
            StoredChunk("a:1", "a.md", 1, "beta rag", [1.1, 2.1]),
            StoredChunk("b:0", "b.md", 0, "gamma", [0.5, 0.1]),
        ],
        rebuild=True,
    )

    assert added == 3
    assert store.count_chunks() == 3
    sources = store.list_sources()
    assert [(source.source, source.chunk_count) for source in sources] == [("a.md", 2), ("b.md", 1)]


def test_vectorstore_queries_chunks(tmp_path):
    store = ChromaVectorStore(tmp_path / "chroma", "collection")
    store.add_chunks(
        [
            StoredChunk("a:0", "a.md", 0, "retrieval augmented generation", [5.0, 2.0]),
            StoredChunk("b:0", "b.md", 0, "database systems", [1.0, 0.0]),
        ],
        rebuild=True,
    )

    results = store.query([5.0, 2.0], top_k=1)

    assert len(results) == 1
    assert results[0].source == "a.md"
