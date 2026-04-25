"""End-to-end test: chunk -> embed -> FAISS store -> search."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings import MultilingualEmbedder
from src.ingestion import Document, chunk_documents
from src.vectorstore import FaissVectorStore


def test_full_pipeline() -> None:
    """Simulate RAG flow: mock documents -> chunk -> embed -> store -> search."""
    MultilingualEmbedder.reset_instance()
    embedder = MultilingualEmbedder()
    store = FaissVectorStore(embedding_dim=embedder.model.get_sentence_embedding_dimension())

    # 1. Mock documents (in real use: load_pdfs_from_directory)
    docs = [
        Document(
            content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "doc1", "page": 1},
        ),
        Document(
            content="Python is a popular programming language for data science.",
            metadata={"source": "doc2", "page": 1},
        ),
        Document(
            content="Natural language processing uses transformers and embeddings.",
            metadata={"source": "doc3", "page": 1},
        ),
    ]

    # 2. Chunk
    chunks = chunk_documents(docs, chunk_size=512, overlap=64)
    assert len(chunks) >= 3

    # 3. Embed (normalize for cosine similarity in FAISS)
    texts = [c.content for c in chunks]
    embeddings = embedder.encode_documents(texts, normalize_embeddings=True)

    # 4. Add to store
    meta = [c.metadata for c in chunks]
    store.add_embeddings(embeddings, meta)

    # 5. Search
    query = "What is machine learning?"
    q_vec = embedder.encode_query(query, normalize_embeddings=True)
    results = store.search(q_vec, k=2)

    assert len(results) >= 1
    assert results[0][0] > 0  # positive similarity
    # Top result should be from doc1 (contains "machine learning")
    top_meta = results[0][1]
    assert top_meta.get("source") == "doc1"


def run_tests() -> None:
    """Run full pipeline test. Downloads embedder model on first run."""
    test_full_pipeline()
    print("  OK test_full_pipeline")
    MultilingualEmbedder.reset_instance()
    print("\nFull pipeline test passed.")


if __name__ == "__main__":
    run_tests()
