"""FastAPI app for DocuMind-RAG query API."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.pipeline import RAGPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default index path
DEFAULT_INDEX_PATH = Path("storage/doc_index.index")

# Global pipeline (initialized in lifespan)
rag_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAGPipeline on startup, clean up on shutdown."""
    global rag_pipeline
    logger.info("Initializing RAG pipeline (index: %s)", DEFAULT_INDEX_PATH)
    try:
        rag_pipeline = RAGPipeline(index_path=DEFAULT_INDEX_PATH)
        logger.info("RAG pipeline initialized successfully")
        yield
    except Exception as e:
        logger.error("Failed to initialize RAG pipeline: %s", e)
        raise
    finally:
        rag_pipeline = None
        logger.info("RAG pipeline shut down")


# --- Pydantic models ---

class QueryInput(BaseModel):
    """Request body for POST /query."""

    question: str = Field(..., min_length=1, description="User question")


class QueryOutput(BaseModel):
    """Response body for POST /query."""

    answer: str = Field(..., description="Generated answer")
    sources: list[dict] = Field(..., description="Source metadata for each chunk")
    scores: list[float] = Field(..., description="Similarity scores for each chunk")


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


@app.post("/query", response_model=QueryOutput, responses={
    400: {"model": ErrorDetail, "description": "Invalid request"},
    503: {"model": ErrorDetail, "description": "Pipeline unavailable"},
})
async def query(input_data: QueryInput) -> QueryOutput:
    """Query the RAG pipeline with a question. Returns answer, sources, and scores."""
    if rag_pipeline is None:
        logger.error("Query attempted but RAG pipeline is not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG pipeline is not available",
        )
    question = input_data.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="question must be non-empty",
        )
    try:
        logger.info("Processing query: %s", question[:80] + ("..." if len(question) > 80 else ""))
        result = await asyncio.to_thread(rag_pipeline.ask, question)
        logger.info("Query completed successfully")
        return QueryOutput(
            answer=result["answer"],
            sources=result["sources"],
            scores=result["scores"],
        )
    except ValueError as e:
        logger.warning("Invalid query: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error processing query: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Error processing query: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
