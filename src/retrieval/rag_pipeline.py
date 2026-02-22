"""RAG pipeline: load index, retrieve chunks, build context. No generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from src.embeddings import MultilingualEmbedder
from src.retrieval.retriever import EmbedderProtocol, Retriever
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
    - Retrieve top-k chunks
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
    ) -> None:
        """Initialize pipeline with a saved FAISS index.

        Args:
            index_path: Path to the index (e.g. storage/doc_index.index).
                Both .index and .meta.json are loaded.
            embedder: Embedder for query encoding. Uses MultilingualEmbedder()
                if not provided. Must match the model used at index build time.
            top_k: Default number of chunks to retrieve per question.
            embedding_dim: Dimension of embeddings. Required for custom embedders
                without .model; otherwise inferred from MultilingualEmbedder.
        """
        self.index_path = Path(index_path)
        self.embedder = embedder or MultilingualEmbedder()
        self.top_k = top_k

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

        Args:
            question: Natural-language user question.
            top_k: Number of chunks to retrieve. Uses self.top_k if not set.

        Returns:
            RAGResult with:
                - context: Concatenated chunk text (newline-separated).
                - sources: List of metadata dicts for each chunk.
                - scores: Similarity scores for each chunk.
        """
        k = top_k if top_k is not None else self.top_k
        results = self._retriever.retrieve(question, k=k)

        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []
        scores: list[float] = []

        for r in results:
            meta = r["metadata"]
            text = meta.get("text", "")
            context_parts.append(text)
            sources.append(meta)
            scores.append(r["score"])

        context = "\n\n".join(context_parts).strip()

        return RAGResult(context=context, sources=sources, scores=scores)
