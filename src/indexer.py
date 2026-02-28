"""Core indexing logic: load PDFs, chunk, embed, build FAISS, and save.

Shared between the CLI build script (build_index.py) and the hot-reload
watcher in main.py so both always produce identical indexes.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class IndexStats:
    """Statistics produced by a completed index build."""

    doc_count: int
    chunk_count: int
    embedding_dim: int
    output_path: Path


def build_index(
    pdf_dir: Path,
    output_path: Path,
    chunk_size: int = 512,
    overlap: int = 64,
    show_progress: bool = False,
) -> IndexStats:
    """Load PDFs, chunk, embed, build a FAISS index, and persist to disk.

    Args:
        pdf_dir: Directory containing PDF files.
        output_path: Destination path for the index (e.g. storage/doc_index.index).
        chunk_size: Token chunk size (default 512).
        overlap: Token overlap between chunks (default 64).
        show_progress: Show tqdm progress bars during embedding.

    Returns:
        IndexStats with doc/chunk counts and embedding dimension.

    Raises:
        ValueError: If no PDFs or chunks are found.
    """
    from src.embeddings import MultilingualEmbedder
    from src.ingestion import chunk_documents, load_pdfs_from_directory
    from src.vectorstore import FaissVectorStore

    # 1. Load
    documents = load_pdfs_from_directory(pdf_dir)
    if not documents:
        raise ValueError(f"No PDFs found in '{pdf_dir}'. Add PDF files and try again.")

    # 2. Chunk
    chunks = chunk_documents(documents, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError("No chunks produced — documents may be empty or unreadable.")

    # 3. Embed
    embedder = MultilingualEmbedder()
    dim_raw = embedder.model.get_sentence_embedding_dimension()
    if not isinstance(dim_raw, int):
        raise TypeError(f"Expected int for embedding dimension, got {type(dim_raw)}")
    dim = dim_raw
    texts = [c.content for c in chunks]
    embeddings = embedder.encode_documents(
        texts, normalize_embeddings=True, show_progress=show_progress
    )

    if embeddings.shape[0] != len(chunks):
        raise RuntimeError(
            f"Embedding count {embeddings.shape[0]} != chunk count {len(chunks)}"
        )

    # 4. Build FAISS store (include chunk text so retrieval has context to pass to LLM)
    store = FaissVectorStore(embedding_dim=dim)
    meta_with_text = [{**c.metadata, "text": c.content} for c in chunks]
    store.add_embeddings(embeddings, meta_with_text)

    # 5. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    store.save(output_path)

    # Save build config for knowledge-base details
    build_info_path = output_path.with_suffix(".build.json")
    build_info = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "strategy": "token-based (tiktoken)",
    }
    with open(build_info_path, "w", encoding="utf-8") as f:
        json.dump(build_info, f, indent=2)

    num_pdfs = len({d.metadata.get("filename") for d in documents})
    logger.info(
        "Index built: %d PDFs, %d chunks, dim=%d → %s",
        num_pdfs, len(chunks), dim, output_path,
    )

    return IndexStats(
        doc_count=num_pdfs,
        chunk_count=len(chunks),
        embedding_dim=dim,
        output_path=output_path,
    )
