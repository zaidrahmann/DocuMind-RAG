"""RAG pipeline: load index, retrieve chunks, optionally rerank, build context. No generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from src.embeddings import MultilingualEmbedder
from src.retrieval.retriever import EmbedderProtocol, Retriever
from src.retrieval.reranker import Reranker
from src.vectorstore import FaissVectorStore


class RAGResult(TypedDict):
    """Structured result from RAG retrieval."""

    context: str
    sources: list[dict[str, Any]]
    scores: list[float]


class RAGPipeline:
    """Pipeline for retrieval-augmented context assembly. No LLM generation.

    Responsibilities:
    - Load saved FAISS index from disk
    - Initialize embedder
    - Accept user question
    - Retrieve top-k (or rerank_top_k when using reranker) chunks
    - Optionally rerank with a cross-encoder and keep top-k for context
    - Combine chunk text into context
    - Return structured result: { context, sources, scores }

    Metadata stored during index build must include "text" for each chunk
    for context assembly. Sources contain the full metadata per chunk
    (filename, page_number, chunk_index, etc.).
    """

    def __init__(
        self,
        index_path: str | Path,
        embedder: EmbedderProtocol | None = None,
        top_k: int = 5,
        embedding_dim: int | None = None,
        reranker: Reranker | None = None,
        rerank_top_k: int = 20,
    ) -> None:
        """Initialize pipeline with a saved FAISS index.

        Args:
            index_path: Path to the index (e.g. storage/doc_index.index).
                Both .index and .meta.json are loaded.
            embedder: Embedder for query encoding. Uses MultilingualEmbedder()
                if not provided. Must match the model used at index build time.
            top_k: Number of chunks to pass to the LLM (after reranking if enabled).
            embedding_dim: Dimension of embeddings. Required for custom embedders
                without .model; otherwise inferred from MultilingualEmbedder.
            reranker: If set, retrieve rerank_top_k candidates, rerank with
                cross-encoder, then keep top_k chunks. Improves precision.
            rerank_top_k: Number of candidates to retrieve for reranking (only
                used when reranker is set). Default 20.
        """
        self.index_path = Path(index_path)
        self.embedder = embedder or MultilingualEmbedder()
        self.top_k = top_k
        self._reranker = reranker
        self._rerank_top_k = rerank_top_k

        if embedding_dim is not None:
            dim = embedding_dim
        elif hasattr(self.embedder, "model") and self.embedder.model is not None:
            dim = self.embedder.model.get_sentence_embedding_dimension()
        else:
            raise ValueError(
                "embedding_dim must be provided when using a custom embedder "
                "without .model attribute"
            )
        self._store = FaissVectorStore(embedding_dim=dim)
        self._store.load(self.index_path)
        self._retriever = Retriever(embedder=self.embedder, store=self._store)

    def run(
        self,
        question: str,
        top_k: int | None = None,
    ) -> RAGResult:
        """Retrieve top-k chunks for a question and assemble context.

        When a reranker is configured: retrieves rerank_top_k candidates,
        reranks with a cross-encoder, then keeps top_k for context. Otherwise
        retrieves top_k directly from the vector store.

        Args:
            question: Natural-language user question.
            top_k: Number of chunks to use for context. Uses self.top_k if not set.

        Returns:
            RAGResult with:
                - context: Concatenated chunk text (newline-separated).
                - sources: List of metadata dicts for each chunk.
                - scores: Similarity/rerank scores for each chunk.
        """
        k = top_k if top_k is not None else self.top_k

        if self._reranker is not None:
            # Retrieve more candidates, rerank, then take top k
            retrieve_k = min(self._rerank_top_k, max(k, 10))
            results = self._retriever.retrieve(question, k=retrieve_k)
            results = self._reranker.rerank(question, results, top_k=k)
        else:
            results = self._retriever.retrieve(question, k=k)

        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        scores_list: list[float] = []

        for r in results:
            meta = r["metadata"]
            text = meta.get("text", "")
            context_parts.append(text)
            sources.append(meta)
            scores_list.append(r["score"])

        context = "\n\n".join(context_parts).strip()

        return RAGResult(context=context, sources=sources, scores=scores_list)
