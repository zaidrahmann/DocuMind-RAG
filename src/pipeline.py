"""Full RAG pipeline: retrieve + generate, return answer and sources."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from src.config import get_settings
from src.exceptions import GeneratorUnavailableError
from src.generation import HFGenerator, OllamaGenerator
from src.retrieval import RAGPipeline as RetrievalPipeline, Reranker
from src.retrieval.retriever import EmbedderProtocol


def _default_generator() -> HFGenerator | OllamaGenerator:
    """Pick generator from config (ollama|hf). Default: ollama."""
    settings = get_settings()
    choice = (settings.documind_generator or "ollama").strip().lower()
    if choice == "hf":
        if not settings.hf_api_key:
            raise GeneratorUnavailableError(
                "HuggingFace generator selected but HF_API_KEY is not set. "
                "Set it in .env or use DOCUMIND_GENERATOR=ollama for local LLM."
            )
        return HFGenerator(
            api_key=settings.hf_api_key,
            model=settings.hf_model or None,
        )
    return OllamaGenerator(
        base_url=settings.ollama_url or None,
        model=settings.ollama_model or None,
    )


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
        use_reranker: bool | None = None,
        rerank_top_k: int = 20,
        reranker: Reranker | None = None,
    ) -> None:
        """Initialize pipeline with index path and optional components.

        Args:
            use_reranker: If True, retrieve rerank_top_k candidates, rerank with
                a cross-encoder, pass top_k to the LLM. Default from env
                DOCUMIND_USE_RERANKER (true) or True.
            rerank_top_k: Candidates to retrieve for reranking when enabled (default 20).
            reranker: Reranker instance; if None and use_reranker, a default Reranker() is used.
        """
        if use_reranker is None:
            use_reranker = get_settings().documind_use_reranker
        effective_reranker = (
            reranker or Reranker(model_name=get_settings().documind_reranker_model or None)
        ) if use_reranker else None
        self._retrieval = RetrievalPipeline(
            index_path=index_path,
            embedder=embedder,
            top_k=top_k,
            embedding_dim=embedding_dim,
            reranker=effective_reranker,
            rerank_top_k=rerank_top_k,
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
