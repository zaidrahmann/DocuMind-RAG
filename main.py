"""FastAPI app for DocuMind-RAG query API."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.config import get_settings
from src.exceptions import DocuMindError, PipelineInitError, ValidationError
from src.logging_config import configure_logging, get_logger
from src.pipeline import RAGPipeline

# Configure logging from settings (after imports so config is available)
_settings = get_settings()
configure_logging(_settings.documind_log_level)
logger = get_logger(__name__)

# Global pipeline (initialized in lifespan)
rag_pipeline: RAGPipeline | None = None

# User-facing message for server errors (no internal details)
SERVICE_UNAVAILABLE_MESSAGE = "Service temporarily unavailable. Please try again later."


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAGPipeline on startup, clean up on shutdown."""
    global rag_pipeline
    settings = get_settings()
    index_path = settings.index_path
    logger.info("Initializing RAG pipeline (index: %s)", index_path)
    try:
        rag_pipeline = RAGPipeline(index_path=index_path)
        logger.info("RAG pipeline initialized successfully")
        yield
    except DocuMindError:
        raise
    except Exception as e:
        logger.exception("Failed to initialize RAG pipeline: %s", e)
        raise PipelineInitError(
            "RAG pipeline failed to start. Check that the index exists and is valid.",
            details=str(e),
        ) from e
    finally:
        rag_pipeline = None
        logger.info("RAG pipeline shut down")


# --- Pydantic models ---


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


class ErrorDetail(BaseModel):
    """Standard error response."""

    detail: str = Field(..., description="Error message")


# --- App ---

app = FastAPI(
    title="DocuMind-RAG API",
    description="RAG query API with retrieval and generation",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthOutput)
async def health() -> HealthOutput:
    """Health check endpoint. Returns ok if the service is running."""
    return HealthOutput(status="ok")


def _validate_question(question: str) -> str:
    """Strip and validate question length. Raises ValidationError if invalid."""
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
    responses={
        400: {"model": ErrorDetail, "description": "Invalid request"},
        503: {"model": ErrorDetail, "description": "Pipeline unavailable"},
    },
)
async def query(input_data: QueryInput) -> QueryOutput:
    """Query the RAG pipeline. Returns answer, sources, and scores."""
    if rag_pipeline is None:
        logger.error("Query attempted but RAG pipeline is not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=SERVICE_UNAVAILABLE_MESSAGE,
        )
    try:
        question = _validate_question(input_data.question)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)

    try:
        logger.info("Processing query: %s", question[:80] + ("..." if len(question) > 80 else ""))
        result = await asyncio.to_thread(rag_pipeline.ask, question)
        logger.info("Query completed successfully")
        return QueryOutput(
            answer=result["answer"],
            sources=result["sources"],
            scores=result["scores"],
        )
    except ValidationError as e:
        logger.warning("Invalid query: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)
    except DocuMindError as e:
        logger.warning("Pipeline error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=SERVICE_UNAVAILABLE_MESSAGE,
        )
    except Exception as e:
        logger.exception("Unexpected error processing query")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=SERVICE_UNAVAILABLE_MESSAGE,
        )


@app.post(
    "/ask",
    response_model=AskOutput,
    responses={
        400: {"model": ErrorDetail, "description": "Invalid request"},
        503: {"model": ErrorDetail, "description": "Pipeline unavailable"},
    },
)
async def ask(input_data: AskInput) -> AskOutput:
    """Ask a question. Returns answer and sources."""
    if rag_pipeline is None:
        logger.error("Ask attempted but RAG pipeline is not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=SERVICE_UNAVAILABLE_MESSAGE,
        )
    try:
        question = _validate_question(input_data.question)
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)

    try:
        logger.info("Processing ask: %s", question[:80] + ("..." if len(question) > 80 else ""))
        result = await asyncio.to_thread(rag_pipeline.ask, question)
        logger.info("Ask completed successfully")
        return AskOutput(
            answer=result["answer"],
            sources=result["sources"],
        )
    except ValidationError as e:
        logger.warning("Invalid ask: %s", e)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.message)
    except DocuMindError as e:
        logger.warning("Pipeline error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=SERVICE_UNAVAILABLE_MESSAGE,
        )
    except Exception as e:
        logger.exception("Unexpected error processing ask")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=SERVICE_UNAVAILABLE_MESSAGE,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
