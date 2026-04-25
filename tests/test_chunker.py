"""Tests for the document chunker."""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import chunk_documents, Document, Chunk


def test_chunk_empty_documents() -> None:
    """Empty document list should produce no chunks."""
    chunks = chunk_documents([])
    assert chunks == []


def test_chunk_single_small_document() -> None:
    """Single short document should produce one chunk."""
    docs = [Document(content="Hello world. Short text.", metadata={"filename": "a.txt", "page_number": 1})]
    chunks = chunk_documents(docs, chunk_size=512, overlap=64)
    assert len(chunks) == 1
    assert chunks[0].content == "Hello world. Short text."
    assert chunks[0].metadata["chunk_index"] == 0
    assert "token_count" in chunks[0].metadata


def test_chunk_large_document_splits() -> None:
    """Long document should be split into multiple chunks with overlap."""
    long_text = " ".join(["word"] * 500)  # Many words to exceed chunk_size tokens
    docs = [Document(content=long_text, metadata={"filename": "big.txt", "page_number": 1})]
    chunks = chunk_documents(docs, chunk_size=128, overlap=16)
    assert len(chunks) > 1
    # Check chunk indices are sequential
    for i, c in enumerate(chunks):
        assert c.metadata["chunk_index"] == i
        assert c.metadata["token_count"] <= 128


def test_chunk_inherits_metadata() -> None:
    """Chunks should inherit document metadata."""
    docs = [Document(content="Test.", metadata={"filename": "x.pdf", "page_number": 5})]
    chunks = chunk_documents(docs)
    assert chunks[0].metadata["filename"] == "x.pdf"
    assert chunks[0].metadata["page_number"] == 5
    assert "chunk_index" in chunks[0].metadata


def test_chunk_invalid_overlap_raises() -> None:
    """overlap >= chunk_size should raise ValueError."""
    docs = [Document(content="x", metadata={})]
    try:
        chunk_documents(docs, chunk_size=64, overlap=64)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "overlap" in str(e).lower()


def run_tests() -> None:
    """Run all chunker tests."""
    tests = [
        test_chunk_empty_documents,
        test_chunk_single_small_document,
        test_chunk_large_document_splits,
        test_chunk_inherits_metadata,
        test_chunk_invalid_overlap_raises,
    ]
    for t in tests:
        t()
        print(f"  OK {t.__name__}")
    print(f"\nAll {len(tests)} chunker tests passed.")


if __name__ == "__main__":
    run_tests()
