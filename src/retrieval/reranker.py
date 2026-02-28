"""Reranker: score (query, chunk) pairs with a cross-encoder for better retrieval quality."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Lazy import to avoid loading the model until first use
_CrossEncoder: type[Any] | None = None


def _get_cross_encoder() -> type[Any]:
    global _CrossEncoder
    if _CrossEncoder is None:
        from sentence_transformers import CrossEncoder
        _CrossEncoder = CrossEncoder
    return _CrossEncoder


class Reranker:
    """Rerank retrieved chunks with a small cross-encoder.

    Retrieve more candidates (e.g. top-20) with bi-encoder/FAISS, then score
    each (question, chunk) pair with a cross-encoder and keep the top-k.
    Improves precision over "vector search only" by modeling query–chunk interaction.
    """

    def __init__(
        self,
        model_name: str | None = None,
    ) -> None:
        """Initialize the reranker.

        Args:
            model_name: HuggingFace model for cross-encoder. Default from
                DOCUMIND_RERANKER_MODEL env, else cross-encoder/ms-marco-MiniLM-L-6-v2.
        """
        self._model_name = (
            model_name
            or os.environ.get("DOCUMIND_RERANKER_MODEL")
            or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            CrossEncoder = _get_cross_encoder()
            logger.info("Loading reranker model: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
            logger.debug("Reranker model loaded")

    def rerank(
        self,
        question: str,
        results: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank retrieval results by cross-encoder (question, chunk) scores.

        Args:
            question: User question.
            results: List of dicts from Retriever.retrieve(), each with
                "score" and "metadata" (metadata must have "text").
            top_k: Number of results to return after reranking.

        Returns:
            Top-k results, reordered by cross-encoder score (descending).
            Each dict has "score" (rerank score) and "metadata".
        """
        if not results:
            return []
        self._ensure_loaded()
        assert self._model is not None  # Type guard after _ensure_loaded()
        pairs = [
            (question, r["metadata"].get("text", "") or "")
            for r in results
        ]
        scores = self._model.predict(pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        elif hasattr(scores, "flatten"):
            scores = scores.flatten().tolist()

        combined = [
            {**r, "score": float(s)}
            for r, s in zip(results, scores, strict=True)
        ]
        combined.sort(key=lambda x: x["score"], reverse=True)
        return combined[:top_k]
