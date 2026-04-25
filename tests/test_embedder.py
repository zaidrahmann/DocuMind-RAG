"""Tests for the MultilingualEmbedder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import MultilingualEmbedder


def test_embedder_singleton() -> None:
    """Embedder should be a singleton (same instance returned)."""
    MultilingualEmbedder.reset_instance()
    a = MultilingualEmbedder()
    b = MultilingualEmbedder()
    assert a is b


def test_encode_documents_empty() -> None:
    """Empty document list should return (0, dim) array."""
    MultilingualEmbedder.reset_instance()
    emb = MultilingualEmbedder()
    out = emb.encode_documents([])
    dim = emb.model.get_sentence_embedding_dimension()
    assert out.shape == (0, dim)
    assert out.dtype == np.float32


def test_encode_documents_batch() -> None:
    """Encode a few texts; output shape (N, dim) and float32."""
    MultilingualEmbedder.reset_instance()
    emb = MultilingualEmbedder()
    texts = ["Hello world.", "Another sentence.", "Third one."]
    out = emb.encode_documents(texts)
    dim = emb.model.get_sentence_embedding_dimension()
    assert out.shape == (3, dim)
    assert out.dtype == np.float32


def test_encode_query() -> None:
    """Encode single query; output 1D vector of correct dim."""
    MultilingualEmbedder.reset_instance()
    emb = MultilingualEmbedder()
    vec = emb.encode_query("What is the meaning of life?")
    dim = emb.model.get_sentence_embedding_dimension()
    assert vec.shape == (dim,)
    assert vec.dtype == np.float32


def test_encode_query_similar_to_documents() -> None:
    """Query similar to a document should have high cosine similarity to that doc's embedding."""
    MultilingualEmbedder.reset_instance()
    emb = MultilingualEmbedder()
    texts = ["Machine learning is a subset of artificial intelligence."]
    doc_vecs = emb.encode_documents(texts, normalize_embeddings=True)
    query_vec = emb.encode_query("What is machine learning?", normalize_embeddings=True)
    sim = np.dot(doc_vecs[0], query_vec)
    assert sim > 0.5  # Should be reasonably similar


def run_tests() -> None:
    """Run all embedder tests. First run downloads the model (~400MB)."""
    tests = [
        test_embedder_singleton,
        test_encode_documents_empty,
        test_encode_documents_batch,
        test_encode_query,
        test_encode_query_similar_to_documents,
    ]
    for t in tests:
        t()
        print(f"  OK {t.__name__}")
    MultilingualEmbedder.reset_instance()
    print(f"\nAll {len(tests)} embedder tests passed.")


if __name__ == "__main__":
    run_tests()
