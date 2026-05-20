# Simple RAG

A local-first retrieval-augmented generation app built with FastAPI, Chroma, and Ollama.

## Features

- Ingest local `.txt`, `.md`, and `.markdown` files from `data/`
- Store embeddings in a local Chroma database
- Query indexed documents through a JSON API or a small browser UI
- Keep model access provider-agnostic with an Ollama implementation
- Upload new documents through the web UI or API

## Quick Start

1. Install dependencies.
2. Start Ollama and pull the models you want to use.
3. Put documents in `data/`.
4. Run the app.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install .[dev]
uvicorn rag_app.main:app --reload
```

The app will be available at `http://127.0.0.1:8000`.

## Environment Variables

- `RAG_DATA_DIR`: directory containing source files. Default: `data`
- `RAG_VECTOR_BACKEND`: vector backend to use. Default: `chroma`
- `RAG_VECTOR_STORE_DIR`: directory for persistent vector store data. Default: `.chroma`
- `RAG_CHROMA_DIR`: legacy fallback for Chroma persistence directory when `RAG_VECTOR_STORE_DIR` is unset
- `RAG_COLLECTION_NAME`: Chroma collection name. Default: `rag_documents`
- `RAG_OLLAMA_BASE_URL`: Ollama base URL. Default: `http://127.0.0.1:11434`
- `RAG_EMBEDDING_MODEL`: Ollama embedding model. Default: `nomic-embed-text`
- `RAG_CHAT_MODEL`: Ollama chat model. Default: `llama3.1`
- `RAG_CHUNK_SIZE`: chunk size in characters. Default: `900`
- `RAG_CHUNK_OVERLAP`: chunk overlap in characters. Default: `150`
- `RAG_TOP_K`: number of retrieved chunks for answers. Default: `4`

## API

- `GET /health`
- `GET /sources`
- `POST /ingest`
- `POST /upload`
- `POST /chat`

## Notes

- v1 supports text and Markdown sources only.
- Supported vector backends currently include `chroma` and `memory`.
- If no relevant context is found, the app returns a bounded response instead of hallucinating from outside the indexed data.


