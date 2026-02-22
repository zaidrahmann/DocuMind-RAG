#!/usr/bin/env python3
"""Build FAISS index from PDFs: load, chunk, embed, and persist.

Run this script to pre-build the vector index. At runtime, load the index
for fast similarity search without re-processing documents.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import get_settings
from src.embeddings import MultilingualEmbedder
from src.ingestion import load_pdfs_from_directory, chunk_documents
from src.logging_config import configure_logging, get_logger
from src.vectorstore import FaissVectorStore


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the build process."""
    level = "DEBUG" if verbose else get_settings().documind_log_level
    configure_logging(level)


def main() -> int:
    """Build index from PDFs and save to disk."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build FAISS index from PDFs")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=settings.documind_pdf_dir,
        help="Directory containing PDF files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=settings.index_path,
        help="Output index path",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Token chunk size (default: 512)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Token overlap between chunks (default: 64)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    logger = get_logger(__name__)

    pdf_dir = args.pdf_dir
    output = args.output
    logger.info("Starting index build: pdf_dir=%s, output=%s", pdf_dir, output)

    # 1. Load PDFs
    documents = load_pdfs_from_directory(pdf_dir)
    if not documents:
        logger.error("No PDFs found in %s. Ensure the directory exists and contains .pdf files.", pdf_dir)
        print(f"Error: No PDFs found in {pdf_dir}", file=sys.stderr)
        return 1

    doc_count = len(documents)
    logger.info("Loaded %d document(s) (pages)", doc_count)

    # 2. Chunk documents
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    if not chunks:
        logger.error("No chunks produced from documents.")
        print("Error: No chunks produced.", file=sys.stderr)
        return 1

    chunk_count = len(chunks)
    logger.info("Produced %d chunk(s)", chunk_count)

    # 3. Initialize embedder and generate embeddings
    embedder = MultilingualEmbedder()
    dim = embedder.model.get_sentence_embedding_dimension()
    texts = [c.content for c in chunks]
    embeddings = embedder.encode_documents(texts, normalize_embeddings=True, show_progress=True)

    if embeddings.shape[0] != chunk_count:
        logger.error("Embedding count mismatch: %d vs %d chunks", embeddings.shape[0], chunk_count)
        return 1

    logger.info("Generated embeddings: shape=%s", embeddings.shape)

    # 4. Build FAISS store and add embeddings (include chunk text for RAG context)
    store = FaissVectorStore(embedding_dim=dim)
    meta_with_text = [{**c.metadata, "text": c.content} for c in chunks]
    store.add_embeddings(embeddings, meta_with_text)

    # 5. Save index
    output.parent.mkdir(parents=True, exist_ok=True)
    store.save(output)
    logger.info("Saved index to %s", output)

    # Print summary counts
    print()
    print("Index build complete.")
    print(f"  Documents (pages): {doc_count}")
    print(f"  Chunks:            {chunk_count}")
    print(f"  Embedding dim:     {dim}")
    print(f"  Output:            {output}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
