"""DocuMind-RAG custom exceptions for clear error handling and API responses."""

from __future__ import annotations


class DocuMindError(Exception):
    """Base exception for DocuMind-RAG. Use for user-facing or logged errors."""

    def __init__(self, message: str, *, details: str | None = None) -> None:
        self.message = message
        self.details = details
        super().__init__(message)


class IndexNotFoundError(DocuMindError):
    """Raised when the FAISS index or metadata file is missing."""


class GeneratorUnavailableError(DocuMindError):
    """Raised when the configured LLM generator cannot be used (e.g. missing API key)."""


class PipelineInitError(DocuMindError):
    """Raised when the RAG pipeline fails to initialize (e.g. index load failure)."""


class ValidationError(DocuMindError):
    """Raised for invalid user input (empty question, out-of-range top_k, etc.)."""
