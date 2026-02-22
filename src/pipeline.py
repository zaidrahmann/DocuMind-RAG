"""Full RAG pipeline: retrieve + generate, return answer and sources."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypedDict

from src.generation import HFGenerator, OllamaGenerator
from src.retrieval import RAGPipeline as RetrievalPipeline
from src.retrieval.retriever import EmbedderProtocol


def _default_generator() -> HFGenerator | OllamaGenerator:
    """Pick generator from DOCUMIND_GENERATOR env (ollama|hf). Default: ollama (HF API unreliable)."""
    choice = (os.environ.get("DOCUMIND_GENERATOR", "ollama") or "ollama").strip().lower()
    if choice == "hf":
        return HFGenerator()
    return OllamaGenerator()


class AskResult(TypedDict):
    """Structured result from RAGPipeline.ask()."""

    answer: str
    sources: list[dict[str, Any]]
    scores: list[float]


class RAGPipeline:
    """Full RAG pipeline: retrieve, combine context, generate answer.

    Flow: 1. retrieve 2. combine context 3. call generator 4. return answer + sources.
    """

    def __init__(
        self,
        index_path: str | Path,
        embedder: EmbedderProtocol | None = None,
        top_k: int = 5,
        embedding_dim: int | None = None,
        generator: HFGenerator | OllamaGenerator | None = None,
    ) -> None:
        """Initialize pipeline with index path and optional components."""
        self._retrieval = RetrievalPipeline(
            index_path=index_path,
            embedder=embedder,
            top_k=top_k,
            embedding_dim=embedding_dim,
        )
        self._generator = generator or _default_generator()

    def ask(self, question: str, top_k: int | None = None) -> AskResult:
        """Retrieve context, generate answer, return answer + sources + scores.

        Args:
            question: Natural-language user question.
            top_k: Optional override for number of chunks to retrieve.

        Returns:
            Dict with keys: answer, sources, scores.
        """
        # 1. retrieve
        rag_result = self._retrieval.run(question, top_k=top_k)
        # 2. combine context (already done in run)
        context = rag_result["context"]
        # 3. call generator
        answer = self._generator.generate(question, context)
        # 4. return answer + sources
        return {
            "answer": answer,
            "sources": rag_result["sources"],
            "scores": rag_result["scores"],
        }
