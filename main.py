"""FastAPI app for DocuMind-RAG query API.

Features:
- POST /ask        — answer + sources
- POST /query      — answer + sources + scores
- GET  /health     — liveness check
- GET  /status     — index stats + watcher state
- Hot reload       — drop a PDF into data/raw_pdfs/ and the index rebuilds
                     automatically, with no server restart required.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from src.config import get_settings
from src.exceptions import DocuMindError, PipelineInitError, ValidationError
from src.indexer import IndexStats, build_index
from src.logging_config import configure_logging, get_logger
from src.pipeline import RAGPipeline

# ── Logging ──────────────────────────────────────────────────────────────────
_settings = get_settings()
configure_logging(_settings.documind_log_level)
logger = get_logger(__name__)

# ── Global state ─────────────────────────────────────────────────────────────
rag_pipeline: RAGPipeline | None = None
_pipeline_lock = asyncio.Lock()          # prevents concurrent rebuilds
SERVICE_UNAVAILABLE_MESSAGE = "Service temporarily unavailable. Please try again later."


@dataclass
class _IndexState:
    """Tracks what the currently loaded index contains and its build history."""
    status: str = "initializing"        # initializing | ready | indexing | error
    doc_count: int = 0
    chunk_count: int = 0
    embedding_dim: int = 0
    last_indexed: str = ""              # ISO-8601 UTC timestamp
    watching_dir: str = ""
    last_error: str = ""


_index_state = _IndexState()


# ── Hot-reload helpers ────────────────────────────────────────────────────────

async def _rebuild_and_swap() -> None:
    """Rebuild the FAISS index from disk and hot-swap the in-memory pipeline.

    Uses _pipeline_lock so only one rebuild runs at a time.
    In-flight /ask and /query requests hold a reference to the old pipeline
    object and complete safely — Python's GIL ensures the pointer swap is atomic.
    """
    global rag_pipeline

    # Skip if a rebuild is already in progress
    if _pipeline_lock.locked():
        logger.info("Re-index skipped — rebuild already in progress")
        return

    settings = get_settings()
    pdf_dir = settings.documind_pdf_dir
    index_path = settings.index_path

    async with _pipeline_lock:
        _index_state.status = "indexing"
        logger.info("Re-indexing %s → %s", pdf_dir, index_path)
        try:
            stats: IndexStats = await asyncio.to_thread(
                build_index,
                pdf_dir,
                index_path,
                show_progress=False,
            )
            new_pipeline = await asyncio.to_thread(RAGPipeline, index_path)
            rag_pipeline = new_pipeline          # atomic pointer swap

            _index_state.status = "ready"
            _index_state.doc_count = stats.doc_count
            _index_state.chunk_count = stats.chunk_count
            _index_state.embedding_dim = stats.embedding_dim
            _index_state.last_indexed = datetime.now(timezone.utc).isoformat(timespec="seconds")
            _index_state.last_error = ""
            logger.info(
                "Hot-reload complete — %d docs, %d chunks", stats.doc_count, stats.chunk_count
            )
        except Exception as exc:
            _index_state.status = "error"
            _index_state.last_error = str(exc)
            logger.exception("Re-index failed: %s", exc)


class _PDFWatchHandler(FileSystemEventHandler):
    """Triggers a pipeline hot-reload whenever a PDF is created, modified, or deleted."""

    # Minimum seconds between rebuilds (debounce for large file copies)
    _COOLDOWN = 5.0

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self._loop = loop
        self._last_triggered: float = 0.0
        self._lock = threading.Lock()

    def _schedule(self, path: str) -> None:
        now = time.monotonic()
        with self._lock:
            if now - self._last_triggered < self._COOLDOWN:
                logger.debug("Watcher debounce — skipping trigger for %s", path)
                return
            self._last_triggered = now
        logger.info("PDF change detected: %s — scheduling re-index", path)
        asyncio.run_coroutine_threadsafe(_rebuild_and_swap(), self._loop)

    def _is_pdf(self, path: str) -> bool:
        return bool(path and not path.endswith("~") and path.lower().endswith(".pdf"))

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_pdf(event.src_path):
            self._schedule(str(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_pdf(event.src_path):
            self._schedule(str(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_pdf(event.src_path):
            self._schedule(str(event.src_path))


# ── Lifespan ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start pipeline + file watcher on startup; clean up on shutdown."""
    global rag_pipeline

    settings = get_settings()
    index_path = settings.index_path
    pdf_dir = settings.documind_pdf_dir

    _index_state.watching_dir = str(pdf_dir)

    # ── Initial pipeline load (optional: start without index if missing) ──────
    logger.info("Loading RAG pipeline from index: %s", index_path)
    if not index_path.exists():
        logger.warning("Index not found at %s — server will start without RAG. Add PDFs to %s and run build_index.py or drop a PDF for auto-build.",
                       index_path, pdf_dir)
        _index_state.status = "no_index"
        rag_pipeline = None
    else:
        try:
            rag_pipeline = RAGPipeline(index_path=index_path)

            # Populate status from the loaded index metadata
            import json as _json
            meta_path = Path(str(index_path)).with_suffix(".meta.json")
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as _f:
                    _meta = _json.load(_f)
                _index_state.doc_count = len({m.get("filename") for m in _meta})
                _index_state.chunk_count = len(_meta)
            _index_state.status = "ready"
            _index_state.last_indexed = datetime.fromtimestamp(
                index_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(timespec="seconds")
            logger.info("RAG pipeline ready")
        except DocuMindError:
            raise
        except Exception as exc:
            logger.exception("Failed to load pipeline: %s", exc)
            raise PipelineInitError(
                "RAG pipeline failed to start. Check that the index exists and is valid.",
                details=str(exc),
            ) from exc

    # ── Start watchdog ────────────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    handler = _PDFWatchHandler(loop)
    observer = Observer()
    pdf_dir.mkdir(parents=True, exist_ok=True)
    observer.schedule(handler, str(pdf_dir), recursive=False)
    observer.start()
    logger.info("Watching %s for new PDFs — index will hot-reload automatically", pdf_dir)

    try:
        yield
    finally:
        observer.stop()
        observer.join()
        rag_pipeline = None
        logger.info("RAG pipeline shut down")


# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryInput(BaseModel):
    """Request body for POST /query."""
    question: str = Field(..., min_length=1, max_length=65536, description="User question")


class QueryOutput(BaseModel):
    """Response body for POST /query."""
    answer: str = Field(..., description="Generated answer")
    sources: list[dict] = Field(..., description="Source metadata for each chunk")
    scores: list[float] = Field(..., description="Similarity scores for each chunk")


class AskInput(BaseModel):
    """Request body for POST /ask."""
    question: str = Field(..., min_length=1, max_length=65536, description="User question")


class AskOutput(BaseModel):
    """Response body for POST /ask (answer + sources only)."""
    answer: str = Field(..., description="Generated answer")
    sources: list[dict] = Field(..., description="Source metadata for each chunk")


class HealthOutput(BaseModel):
    """Response body for GET /health."""
    status: str = Field(..., description="Health status")


class StatusOutput(BaseModel):
    """Response body for GET /status — index stats and watcher info."""
    status: str = Field(..., description="Pipeline status: initializing | ready | indexing | error | no_index")
    doc_count: int = Field(..., description="Number of source documents in the current index")
    chunk_count: int = Field(..., description="Number of indexed chunks")
    embedding_dim: int = Field(..., description="Embedding vector dimension")
    last_indexed: str = Field(..., description="ISO-8601 UTC timestamp of last successful index build")
    watching_dir: str = Field(..., description="Directory being watched for new PDFs")
    last_error: str = Field(..., description="Last indexing error message (empty if none)")


class DocumentStats(BaseModel):
    """Per-document stats for knowledge base details."""
    filename: str = Field(..., description="PDF filename")
    page_count: int = Field(..., description="Number of pages in the document")
    chunk_count: int = Field(..., description="Number of chunks from this document")


class KnowledgeBaseOutput(BaseModel):
    """Response body for GET /knowledge-base — detailed index statistics."""
    total_document_count: int = Field(..., description="Total number of PDF documents")
    total_chunk_count: int = Field(..., description="Total number of indexed chunks")
    documents: list[DocumentStats] = Field(..., description="Per-document breakdown")
    chunking_strategy: str = Field(..., description="Chunking method used")
    chunk_size: int = Field(..., description="Chunk size in tokens")
    overlap: int = Field(..., description="Overlap between chunks in tokens")


class ErrorDetail(BaseModel):
    """Standard error response."""
    detail: str = Field(..., description="Error message")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="DocuMind-RAG API",
    description=(
        "Production-ready RAG query API with hot-reload indexing.\n\n"
        "Drop a PDF into the watched directory and the index updates automatically."
    ),
    version="1.1.0",
    lifespan=lifespan,
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs", status_code=status.HTTP_302_FOUND)


@app.get("/health", response_model=HealthOutput, tags=["Ops"])
async def health() -> HealthOutput:
    """Liveness check. Returns ok if the server is running."""
    return HealthOutput(status="ok")


@app.get("/status", response_model=StatusOutput, tags=["Ops"])
async def index_status() -> StatusOutput:
    """Index and watcher status — doc count, last indexed time, watch directory."""
    return StatusOutput(
        status=_index_state.status,
        doc_count=_index_state.doc_count,
        chunk_count=_index_state.chunk_count,
        embedding_dim=_index_state.embedding_dim,
        last_indexed=_index_state.last_indexed,
        watching_dir=_index_state.watching_dir,
        last_error=_index_state.last_error,
    )


def _get_knowledge_base_details() -> KnowledgeBaseOutput:
    """Compute knowledge base stats from index metadata and build info."""
    settings = get_settings()
    index_path = settings.index_path
    meta_path = index_path.with_suffix(".meta.json")
    build_path = index_path.with_suffix(".build.json")

    chunk_size = 512
    overlap = 64
    chunking_strategy = "token-based (tiktoken)"
    if build_path.exists():
        try:
            with open(build_path, encoding="utf-8") as f:
                build = json.load(f)
            chunk_size = build.get("chunk_size", chunk_size)
            overlap = build.get("overlap", overlap)
            chunking_strategy = build.get("strategy", chunking_strategy)
        except Exception:
            pass

    documents: list[DocumentStats] = []
    total_chunk_count = 0
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            by_file: dict[str, dict] = {}
            for m in meta:
                fn = m.get("filename") or "unknown"
                if fn not in by_file:
                    by_file[fn] = {"pages": set(), "chunks": 0}
                by_file[fn]["pages"].add(m.get("page_number") or 0)
                by_file[fn]["chunks"] += 1
            for fn, stats in sorted(by_file.items()):
                page_count = len(stats["pages"]) if stats["pages"] else 0
                chunk_count = stats["chunks"]
                total_chunk_count += chunk_count
                documents.append(
                    DocumentStats(filename=fn, page_count=page_count, chunk_count=chunk_count)
                )
        except Exception:
            pass

    return KnowledgeBaseOutput(
        total_document_count=len(documents),
        total_chunk_count=total_chunk_count or _index_state.chunk_count,
        documents=documents,
        chunking_strategy=chunking_strategy,
        chunk_size=chunk_size,
        overlap=overlap,
    )


@app.get("/knowledge-base", response_model=KnowledgeBaseOutput, tags=["Ops"])
async def knowledge_base() -> KnowledgeBaseOutput:
    """Detailed knowledge base statistics: documents, pages, chunks, chunking config."""
    return _get_knowledge_base_details()


def _validate_question(question: str) -> str:
    q = question.strip()
    if not q:
        raise ValidationError("question must be non-empty")
    max_len = get_settings().documind_max_question_length
    if len(q) > max_len:
        raise ValidationError(f"question must be at most {max_len} characters")
    return q


@app.post(
    "/query",
    response_model=QueryOutput,
    tags=["RAG"],
    responses={
        400: {"model": ErrorDetail, "description": "Invalid request"},
        503: {"model": ErrorDetail, "description": "Pipeline unavailable"},
    },
)
async def query(input_data: QueryInput) -> QueryOutput:
    """Query the RAG pipeline. Returns answer, sources, and similarity scores."""
    if rag_pipeline is None:
        logger.error("Query attempted but pipeline is not initialized")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=SERVICE_UNAVAILABLE_MESSAGE)
    try:
        question = _validate_question(input_data.question)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message) from e
    try:
        logger.info("Query: %s", question[:80] + ("..." if len(question) > 80 else ""))
        result = await asyncio.to_thread(rag_pipeline.ask, question)
        logger.info("Query complete")
        return QueryOutput(answer=result["answer"], sources=result["sources"], scores=result["scores"])
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message) from e
    except DocuMindError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=SERVICE_UNAVAILABLE_MESSAGE) from e
    except Exception as e:
        logger.exception("Unexpected error processing query")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=SERVICE_UNAVAILABLE_MESSAGE) from e


@app.post(
    "/ask",
    response_model=AskOutput,
    tags=["RAG"],
    responses={
        400: {"model": ErrorDetail, "description": "Invalid request"},
        503: {"model": ErrorDetail, "description": "Pipeline unavailable"},
    },
)
async def ask(input_data: AskInput) -> AskOutput:
    """Ask a question. Returns answer and sources (no scores)."""
    if rag_pipeline is None:
        logger.error("Ask attempted but pipeline is not initialized")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=SERVICE_UNAVAILABLE_MESSAGE)
    try:
        question = _validate_question(input_data.question)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message) from e
    try:
        logger.info("Ask: %s", question[:80] + ("..." if len(question) > 80 else ""))
        result = await asyncio.to_thread(rag_pipeline.ask, question)
        logger.info("Ask complete")
        return AskOutput(answer=result["answer"], sources=result["sources"])
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message) from e
    except DocuMindError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=SERVICE_UNAVAILABLE_MESSAGE) from e
    except Exception as e:
        logger.exception("Unexpected error processing ask")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=SERVICE_UNAVAILABLE_MESSAGE) from e


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import uvicorn

    settings = get_settings()
    port = settings.documind_port
    # reload=False is required when running the file watcher — uvicorn's reloader
    # spawns a subprocess on every .py change which would kill the watchdog observer.
    logger.info("Starting DocuMind API on 127.0.0.1:%s (set DOCUMIND_PORT to use another port)", port)
    try:
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=port,
            reload=False,
        )
    except OSError as e:
        in_use = (
            getattr(e, "winerror", None) == 10048
            or getattr(e, "errno", None) == 98
            or "address already in use" in str(e).lower()
        )
        if in_use:
            logger.error(
                "Port %s is already in use. Stop the other process using it, or set DOCUMIND_PORT to another port (e.g. DOCUMIND_PORT=8001 in .env).",
                port,
            )
            sys.exit(1)
        raise
