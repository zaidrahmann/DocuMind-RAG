"""CLI entry point: python -m documind ask "Your question here"."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on path so "src" can be imported when running from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import get_settings
from src.exceptions import IndexNotFoundError, DocuMindError
from src.pipeline import RAGPipeline


def _run_ask(question: str, index_path: Path) -> int:
    if not index_path.exists():
        raise IndexNotFoundError(
            f"Index not found at {index_path.resolve()}.",
            details="Build it first: python build_index.py --pdf-dir data/raw_pdfs --output storage/doc_index.index",
        )
    pipeline = RAGPipeline(index_path=index_path)
    result = pipeline.ask(question)
    print(result["answer"])
    if result["sources"]:
        print("\n--- Sources ---")
        for i, src in enumerate(result["sources"], 1):
            title = src.get("title") or src.get("source") or "Chunk"
            page = src.get("page")
            part = f" (page {page})" if page is not None else ""
            print(f"  {i}. {title}{part}")
    return 0


def main() -> int:
    settings = get_settings()
    default_index = settings.index_path
    parser = argparse.ArgumentParser(
        prog="documind",
        description="Ask questions over your indexed documents. Use: python -m documind ask \"Your question\"",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=default_index,
        help=f"Path to FAISS index (default: {default_index})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command")

    ask_parser = subparsers.add_parser("ask", help="Ask a question and print answer + sources")
    ask_parser.add_argument(
        "question",
        type=str,
        nargs="+",
        help="Question (words are joined if passed separately)",
    )

    args = parser.parse_args()
    if args.command == "ask":
        question = " ".join(args.question).strip()
        if not question:
            print("Error: question must be non-empty.", file=sys.stderr)
            return 1
        try:
            return _run_ask(question, args.index)
        except IndexNotFoundError as e:
            print(f"Error: {e.message}", file=sys.stderr)
            if e.details:
                print(e.details, file=sys.stderr)
            return 1
        except DocuMindError as e:
            print(f"Error: {e.message}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
