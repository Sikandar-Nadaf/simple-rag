from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from rag_app.config import Settings
from rag_app.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    SourceInfo,
    UploadResponse,
)
from rag_app.providers.ollama import OllamaProvider
from rag_app.services import RagService
from rag_app.vectorstore import ChromaVectorStore


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def build_service(settings: Settings) -> RagService:
    settings.ensure_directories()
    provider = OllamaProvider(
        base_url=settings.ollama_base_url,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
    )
    vector_store = ChromaVectorStore(
        persist_directory=settings.chroma_dir,
        collection_name=settings.collection_name,
    )
    return RagService(
        settings=settings,
        embedding_provider=provider,
        chat_provider=provider,
        vector_store=vector_store,
    )


def create_app(service: RagService | None = None, settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="First RAG", version="0.1.0")
    resolved_settings = settings or Settings.from_env()
    app.state.service = service or build_service(resolved_settings)
    app.state.settings = resolved_settings

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            request,
            "index.html",
            {"request": request, "settings": app.state.settings.public_dict()},
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        rag_service: RagService = app.state.service
        sources = rag_service.sources()
        return HealthResponse(
            status="ok",
            indexed_documents=len(sources),
            indexed_chunks=rag_service.vector_store.count_chunks(),
            settings=app.state.settings.public_dict(),
        )

    @app.get("/sources", response_model=list[SourceInfo])
    def list_sources() -> list[SourceInfo]:
        rag_service: RagService = app.state.service
        return rag_service.sources()

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(payload: IngestRequest) -> IngestResponse:
        rag_service: RagService = app.state.service
        try:
            return rag_service.ingest(rebuild=payload.rebuild)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Ingestion failed: {exc}") from exc

    @app.post("/upload", response_model=UploadResponse)
    async def upload(file: UploadFile = File(...)) -> UploadResponse:
        rag_service: RagService = app.state.service
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".txt", ".md", ".markdown"}:
            raise HTTPException(status_code=400, detail="Only text and Markdown files are supported")
        content = await file.read()
        target = rag_service.save_upload(file.filename, content)
        return UploadResponse(filename=target.name, path=str(target))

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest) -> ChatResponse:
        rag_service: RagService = app.state.service
        try:
            return rag_service.chat(payload.question, top_k=payload.top_k)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Chat failed: {exc}") from exc

    return app


app = create_app()
