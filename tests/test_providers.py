import pytest

from rag_app.providers.factory import create_chat_provider, create_embedding_provider
from rag_app.providers.ollama import OllamaProvider


def test_create_embedding_provider_supports_simple_backend():
    provider = create_embedding_provider(
        "simple",
        base_url="http://localhost:11434",
        embedding_model="unused",
        chat_model="unused",
    )

    vectors = provider.embed_texts(["rag", "hello world"])

    assert len(vectors) == 2
    assert all(len(vector) == 4 for vector in vectors)


def test_create_embedding_provider_supports_ollama_backend():
    provider = create_embedding_provider(
        "ollama",
        base_url="http://localhost:11434",
        embedding_model="embed-model",
        chat_model="chat-model",
    )

    assert isinstance(provider, OllamaProvider)


def test_create_chat_provider_supports_echo_backend():
    provider = create_chat_provider(
        "echo",
        base_url="http://localhost:11434",
        embedding_model="unused",
        chat_model="unused",
    )

    response = provider.chat(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is RAG?"},
        ]
    )

    assert "What is RAG?" in response


@pytest.mark.parametrize(
    ("factory", "backend"),
    [
        (create_embedding_provider, "unknown"),
        (create_chat_provider, "unknown"),
    ],
)
def test_provider_factories_reject_unknown_backends(factory, backend):
    with pytest.raises(ValueError):
        factory(
            backend,
            base_url="http://localhost:11434",
            embedding_model="unused",
            chat_model="unused",
        )
