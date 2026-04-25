"""Tests for RAGPipeline."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieval import RAGPipeline, RAGResult
from src.vectorstore import FaissVectorStore

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class FakeEmbedder:
    """Minimal embedder for tests."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.model = type("Model", (), {"get_sentence_embedding_dimension": lambda self: dim})()

    def encode_query(
        self, text: str, normalize_embeddings: bool = False, **kwargs: object
    ) -> np.ndarray:
        v = np.array([1.0, 0.1, 0.0, 0.0], dtype=np.float32)
        if len(v) != self.dim:
            v = np.resize(v, self.dim)
        if normalize_embeddings:
            n = np.linalg.norm(v)
            v = v / n if n > 0 else v
        return v


def test_rag_pipeline_returns_structured_result() -> None:
    """RAGPipeline.run returns RAGResult with context, sources, scores."""
    store = FaissVectorStore(embedding_dim=4)
    store.add_embeddings(
        np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]], dtype=np.float32),
        [
            {"text": "Chunk one content.", "filename": "a.pdf", "page_number": 1},
            {"text": "Chunk two content.", "filename": "b.pdf", "page_number": 2},
        ],
    )

    tmp = PROJECT_ROOT / "storage" / "_test_rag_pipeline"
    tmp.mkdir(parents=True, exist_ok=True)
    idx = tmp / "test_index"
    try:
        store.save(idx)
        pipeline = RAGPipeline(
            index_path=idx,
            embedder=FakeEmbedder(dim=4),
            top_k=2,
        )
        result = pipeline.run("What is chunk one?")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    assert isinstance(result, dict)
    assert "context" in result
    assert "sources" in result
    assert "scores" in result
    assert isinstance(result["context"], str)
    assert isinstance(result["sources"], list)
    assert isinstance(result["scores"], list)
    assert "Chunk one content." in result["context"]
    assert len(result["sources"]) == 2
    assert len(result["scores"]) == 2


def test_rag_pipeline_respects_top_k() -> None:
    """RAGPipeline.run respects top_k parameter."""
    store = FaissVectorStore(embedding_dim=4)
    store.add_embeddings(
        np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]], dtype=np.float32),
        [{"text": "A"}, {"text": "B"}, {"text": "C"}],
    )

    tmp = PROJECT_ROOT / "storage" / "_test_rag_pipeline_k"
    tmp.mkdir(parents=True, exist_ok=True)
    idx = tmp / "test_index"
    try:
        store.save(idx)
        pipeline = RAGPipeline(index_path=idx, embedder=FakeEmbedder(dim=4), top_k=5)
        result1 = pipeline.run("q", top_k=1)
        result2 = pipeline.run("q", top_k=2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    assert len(result1["sources"]) == 1
    assert len(result2["sources"]) == 2
    assert len(result1["scores"]) == 1
    assert len(result2["scores"]) == 2


def run_tests() -> None:
    """Run all RAGPipeline tests."""
    tests = [
        test_rag_pipeline_returns_structured_result,
        test_rag_pipeline_respects_top_k,
    ]
    for t in tests:
        t()
        print(f"  OK {t.__name__}")
    print("\nAll RAGPipeline tests passed.")


if __name__ == "__main__":
    run_tests()
