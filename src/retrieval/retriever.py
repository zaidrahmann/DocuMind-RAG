"""Retriever: embed query, search FAISS, return scored results."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from src.vectorstore import FaissVectorStore

if TYPE_CHECKING:
    import numpy as np


class EmbedderProtocol(Protocol):
    """Protocol for embedders with encode_query."""

    def encode_query(
        self, text: str, normalize_embeddings: bool = False, **kwargs: Any
    ) -> np.ndarray: ...


class Retriever:
    """Retrieves relevant chunks given a question via embedding + FAISS search."""

    def __init__(
        self,
        embedder: EmbedderProtocol,
        store: FaissVectorStore,
    ) -> None:
        """Initialize with an embedder and FAISS vector store.

        Args:
            embedder: Must provide encode_query(text, normalize_embeddings=...).
            store: FaissVectorStore with search(query_vector, k).
        """
        self.embedder = embedder
        self.store = store

    def retrieve(self, question: str, k: int = 5) -> list[dict[str, Any]]:
        """Retrieve top-k chunks for the given question.

        Process:
        1. Embed query (normalize=True for cosine similarity)
        2. Search FAISS
        3. Return structured list [{score, metadata}, ...]

        Args:
            question: Natural-language query.
            k: Number of results to return.

        Returns:
            List of dicts: [{"score": float, "metadata": dict}, ...]
        """
        query_embedding = self.embedder.encode_query(
            question, normalize_embeddings=True
        )
        raw = self.store.search(query_embedding, k=k)
        return [{"score": s, "metadata": m} for s, m in raw]
