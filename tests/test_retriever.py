"""Tests for the Retriever."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import Retriever
from src.vectorstore import FaissVectorStore


class FakeEmbedder:
    """Minimal embedder for tests; returns fixed query vector."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim

    def encode_query(
        self, text: str, normalize_embeddings: bool = False, **kwargs: object
    ) -> np.ndarray:
        """Return a fixed 1D vector; first dim biased for similarity to index vec 0."""
        v = np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32)
        if len(v) != self.dim:
            v = np.resize(v, self.dim)
        if normalize_embeddings:
            n = np.linalg.norm(v)
            v = v / n if n > 0 else v
        return v


def test_retriever_returns_structured_list() -> None:
    """Retrieve returns [{"score": float, "metadata": dict}, ...]."""
    store = FaissVectorStore(embedding_dim=4)
    store.build_index()
    emb = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]], dtype=np.float32)
    meta = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    store.add_embeddings(emb, meta)

    embedder = FakeEmbedder(dim=4)
    retriever = Retriever(embedder=embedder, store=store)
    results = retriever.retrieve("foo", k=2)

    assert len(results) == 2
    for r in results:
        assert "score" in r
        assert "metadata" in r
        assert isinstance(r["score"], (int, float))
        assert isinstance(r["metadata"], dict)
    assert results[0]["metadata"]["id"] == "a"
    assert results[0]["score"] >= results[1]["score"]


def test_retriever_respects_k() -> None:
    """Retrieve returns exactly k results (or fewer if index is smaller)."""
    store = FaissVectorStore(embedding_dim=4)
    store.build_index()
    emb = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=np.float32)
    store.add_embeddings(emb, [{"x": 1}, {"x": 2}])

    retriever = Retriever(embedder=FakeEmbedder(dim=4), store=store)
    assert len(retriever.retrieve("q", k=1)) == 1
    assert len(retriever.retrieve("q", k=5)) == 2


def test_retriever_empty_store_returns_empty() -> None:
    """Retrieve on empty store returns empty list."""
    store = FaissVectorStore(embedding_dim=4)
    store.build_index()
    retriever = Retriever(embedder=FakeEmbedder(dim=4), store=store)
    assert retriever.retrieve("anything", k=5) == []


def test_retriever_calls_embedder_with_normalize() -> None:
    """Retriever passes normalize_embeddings=True to embedder."""
    store = FaissVectorStore(embedding_dim=4)
    store.add_embeddings(
        np.array([[1.0, 0, 0, 0]], dtype=np.float32), [{"id": 0}]
    )

    class TrackedEmbedder(FakeEmbedder):
        def encode_query(self, text: str, normalize_embeddings: bool = False, **kwargs: object) -> np.ndarray:
            self.last_normalize = normalize_embeddings
            return super().encode_query(text, normalize_embeddings=normalize_embeddings, **kwargs)

    embedder = TrackedEmbedder(dim=4)
    retriever = Retriever(embedder=embedder, store=store)
    retriever.retrieve("query", k=1)
    assert embedder.last_normalize is True


def run_tests() -> None:
    """Run all Retriever tests."""
    tests = [
        test_retriever_returns_structured_list,
        test_retriever_respects_k,
        test_retriever_empty_store_returns_empty,
        test_retriever_calls_embedder_with_normalize,
    ]
    for t in tests:
        t()
        print(f"  OK {t.__name__}")
    print(f"\nAll {len(tests)} Retriever tests passed.")


if __name__ == "__main__":
    run_tests()
