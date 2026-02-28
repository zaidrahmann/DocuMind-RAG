"""Centralized configuration via environment variables (Pydantic Settings)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment and .env.

    Env vars use uppercase field names, e.g. DOCUMIND_INDEX_PATH, OLLAMA_URL.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths
    documind_index_path: Path = Field(
        default=Path("storage/doc_index.index"),
        description="Path to FAISS index (and .meta.json).",
    )
    documind_pdf_dir: Path = Field(
        default=Path("data/raw_pdfs"),
        description="Default directory for PDFs when building index.",
    )

    # Generator: ollama | hf
    documind_generator: str = Field(default="ollama", description="LLM backend: ollama or hf.")

    # Ollama
    ollama_url: str = Field(default="http://localhost:11434", description="Ollama server URL.")
    ollama_model: str = Field(default="llama3", description="Ollama model name.")

    # HuggingFace
    hf_api_key: str = Field(default="", description="HuggingFace API key for inference.")
    hf_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="HuggingFace model for inference.",
    )

    # Reranking
    documind_use_reranker: bool = Field(default=True, description="Use cross-encoder reranking.")
    documind_reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking.",
    )

    # API / server
    documind_api_url: str = Field(
        default="http://localhost:8000",
        description="Base URL for DocuMind API (used by Gradio UI).",
    )
    documind_log_level: str = Field(default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR).")

    # Limits (input validation)
    documind_max_question_length: int = Field(
        default=4096,
        ge=1,
        le=65536,
        description="Max character length for question input.",
    )
    documind_max_top_k: int = Field(default=50, ge=1, le=100, description="Max top_k for retrieval.")

    @property
    def index_path(self) -> Path:
        """Convenience alias for documind_index_path."""
        return self.documind_index_path

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment (with .env)."""
        return cls()


def get_settings() -> Settings:
    """Return application settings (singleton-style; create once per process)."""
    return Settings.from_env()
