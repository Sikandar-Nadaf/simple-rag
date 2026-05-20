from fastapi.testclient import TestClient

from rag_app.main import create_app


def test_health_and_sources_endpoints(fake_service):
    client = TestClient(create_app(service=fake_service, settings=fake_service.settings))

    health = client.get("/health")
    sources = client.get("/sources")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert sources.status_code == 200
    assert sources.json() == []


def test_ingest_and_chat_flow(fake_service):
    (fake_service.settings.data_dir / "intro.md").write_text(
        "RAG stands for retrieval augmented generation.", encoding="utf-8"
    )
    client = TestClient(create_app(service=fake_service, settings=fake_service.settings))

    ingest_response = client.post("/ingest", json={"rebuild": True})
    chat_response = client.post("/chat", json={"question": "What does RAG stand for?"})

    assert ingest_response.status_code == 200
    assert ingest_response.json()["ingested_files"] == 1
    assert chat_response.status_code == 200
    payload = chat_response.json()
    assert "stubbed answer" in payload["answer"]
    assert payload["retrieved_chunks"][0]["source"] == "intro.md"


def test_chat_handles_empty_index(fake_service):
    client = TestClient(create_app(service=fake_service, settings=fake_service.settings))

    response = client.post("/chat", json={"question": "Anything there?"})

    assert response.status_code == 200
    assert response.json()["retrieved_chunks"] == []
    assert "No documents have been indexed yet" in response.json()["answer"]
