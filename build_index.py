#!/usr/bin/env python3
"""Build FAISS index from PDFs: load, chunk, embed, and persist.

Run this script to pre-build the vector index. At runtime, the server
loads the index for fast similarity search — no reprocessing needed.

For automatic hot-reload when PDFs change, just drop files into the
watched directory while python main.py is running.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import get_settings
from src.indexer import build_index
from src.logging_config import configure_logging, get_logger


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

    level = "DEBUG" if args.verbose else settings.documind_log_level
    configure_logging(level)
    logger = get_logger(__name__)
    logger.info("Starting index build: pdf_dir=%s, output=%s", args.pdf_dir, args.output)

    try:
        stats = build_index(
            pdf_dir=args.pdf_dir,
            output_path=args.output,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            show_progress=True,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print()
    print("Index build complete.")
    print(f"  Documents (pages): {stats.doc_count}")
    print(f"  Chunks:            {stats.chunk_count}")
    print(f"  Embedding dim:     {stats.embedding_dim}")
    print(f"  Output:            {stats.output_path}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
