"""Tests for the FAISS vector store."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.vectorstore import FaissVectorStore


def test_store_init() -> None:
    """Store initializes with embedding_dim; build_index creates empty index."""
    store = FaissVectorStore(embedding_dim=128)
    assert store.embedding_dim == 128
    store.build_index()
    assert store._index is not None
    assert store._index.ntotal == 0


def test_add_and_search() -> None:
    """Add embeddings and search; top result should match the query."""
    store = FaissVectorStore(embedding_dim=4)
    store.build_index()
    # Use simple vectors - store normalizes internally
    emb = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]], dtype=np.float32)
    meta = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    store.add_embeddings(emb, meta)
    # Query similar to first vector
    q = np.array([[1.0, 0.1, 0, 0]], dtype=np.float32)
    results = store.search(q, k=2)
    assert len(results) == 2
    assert results[0][1]["id"] == "a"
    assert results[0][0] > results[1][0]


def test_search_empty_returns_empty() -> None:
    """Search on empty index returns empty list."""
    store = FaissVectorStore(embedding_dim=4)
    store.build_index()
    q = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
    assert store.search(q, k=5) == []


def test_save_and_load() -> None:
    """Save index to disk and load; search results should match."""
    store = FaissVectorStore(embedding_dim=8)
    store.build_index()
    rng = np.random.default_rng(42)
    emb = rng.random((5, 8), dtype=np.float32)
    meta = [{"idx": i} for i in range(5)]
    store.add_embeddings(emb, meta)

    d = Path(__file__).parent / "_tmp_faiss"
    d.mkdir(exist_ok=True)
    path = d / "test_index"
    try:
        store.save(path)
        loaded = FaissVectorStore(embedding_dim=8)
        loaded.load(path)
        q = emb[2:3]  # Query with third vector
        orig = store.search(q, k=2)
        reloaded = loaded.search(q, k=2)
        assert orig[0][1] == reloaded[0][1]
        assert abs(orig[0][0] - reloaded[0][0]) < 1e-5
    finally:
        for f in d.glob("*"):
            f.unlink()
        d.rmdir()


def test_add_auto_builds_index() -> None:
    """add_embeddings without prior build_index should auto-build."""
    store = FaissVectorStore(embedding_dim=4)
    emb = np.array([[1.0, 0, 0, 0]], dtype=np.float32)
    store.add_embeddings(emb, [{"x": 1}])
    assert store._index is not None
    assert store._index.ntotal == 1


def test_wrong_dim_raises() -> None:
    """Adding embeddings with wrong dim should raise ValueError."""
    store = FaissVectorStore(embedding_dim=4)
    store.build_index()
    wrong = np.array([[1.0, 0, 0]], dtype=np.float32)  # dim 3
    try:
        store.add_embeddings(wrong, [{}])
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "dim" in str(e).lower() or "embedding" in str(e).lower()


def run_tests() -> None:
    """Run all FAISS store tests."""
    tests = [
        test_store_init,
        test_add_and_search,
        test_search_empty_returns_empty,
        test_save_and_load,
        test_add_auto_builds_index,
        test_wrong_dim_raises,
    ]
    for t in tests:
        t()
        print(f"  OK {t.__name__}")
    print(f"\nAll {len(tests)} FAISS store tests passed.")


if __name__ == "__main__":
    run_tests()
