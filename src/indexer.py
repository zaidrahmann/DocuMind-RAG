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


def _load_build_config(output_path: Path) -> tuple[int, int]:
    """Load chunking config from .build.json with safe defaults."""
    chunk_size = 512
    overlap = 64
    build_info_path = output_path.with_suffix(".build.json")
    if build_info_path.exists():
        try:
            with open(build_info_path, encoding="utf-8") as f:
                build_info = json.load(f)
            chunk_size = int(build_info.get("chunk_size", chunk_size))
            overlap = int(build_info.get("overlap", overlap))
        except Exception:
            logger.warning("Failed to parse %s; using defaults", build_info_path)
    return chunk_size, overlap


def _compute_stats_from_metadata(
    metadata: list[dict],
    output_path: Path,
    embedding_dim: int,
) -> IndexStats:
    doc_count = len({m.get("filename") for m in metadata if m.get("filename")})
    return IndexStats(
        doc_count=doc_count,
        chunk_count=len(metadata),
        embedding_dim=embedding_dim,
        output_path=output_path,
    )


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


def upsert_pdf_in_index(
    pdf_path: Path,
    output_path: Path,
    show_progress: bool = False,
) -> IndexStats:
    """Upsert a single PDF into an existing index without full rebuild.

    Replaces prior chunks for the same filename, then adds the latest chunks.
    """
    from src.embeddings import MultilingualEmbedder
    from src.ingestion import chunk_documents
    from src.ingestion.loader import _load_single_pdf
    from src.vectorstore import FaissVectorStore

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    embedder = MultilingualEmbedder()
    dim_raw = embedder.model.get_sentence_embedding_dimension()
    if not isinstance(dim_raw, int):
        raise TypeError(f"Expected int for embedding dimension, got {type(dim_raw)}")
    dim = dim_raw

    store = FaissVectorStore(embedding_dim=dim)
    if output_path.exists():
        store.load(output_path)
    else:
        store.build_index()

    filename = pdf_path.name
    removed = store.remove_by_filename(filename)
    if removed:
        logger.info("Incremental upsert: removed %d old chunks for %s", removed, filename)

    chunk_size, overlap = _load_build_config(output_path)
    docs = _load_single_pdf(pdf_path)
    chunks = chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
    if chunks:
        texts = [c.content for c in chunks]
        embeddings = embedder.encode_documents(
            texts, normalize_embeddings=True, show_progress=show_progress
        )
        meta_with_text = [{**c.metadata, "text": c.content} for c in chunks]
        store.add_embeddings(embeddings, meta_with_text)
        logger.info("Incremental upsert: added %d chunks for %s", len(chunks), filename)
    else:
        logger.warning("Incremental upsert: no chunks produced for %s", filename)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    store.save(output_path)
    _, metadata = store.export_all()
    return _compute_stats_from_metadata(metadata, output_path, dim)


def delete_pdf_from_index(
    filename: str,
    output_path: Path,
) -> IndexStats:
    """Delete all chunks for one PDF filename from an existing index."""
    from src.embeddings import MultilingualEmbedder
    from src.vectorstore import FaissVectorStore

    embedder = MultilingualEmbedder()
    dim_raw = embedder.model.get_sentence_embedding_dimension()
    if not isinstance(dim_raw, int):
        raise TypeError(f"Expected int for embedding dimension, got {type(dim_raw)}")
    dim = dim_raw

    store = FaissVectorStore(embedding_dim=dim)
    if output_path.exists():
        store.load(output_path)
    else:
        store.build_index()

    removed = store.remove_by_filename(filename)
    logger.info("Incremental delete: removed %d chunks for %s", removed, filename)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    store.save(output_path)
    _, metadata = store.export_all()
    return _compute_stats_from_metadata(metadata, output_path, dim)
