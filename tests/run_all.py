"""Run all test scripts to verify DocuMind-RAG components."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    """Run each test module and report results."""
    modules = [
        ("Chunker", "tests.test_chunker"),
        ("Loader", "tests.test_loader"),
        ("FAISS Store", "tests.test_faiss_store"),
        ("Retriever", "tests.test_retriever"),
        ("Embedder", "tests.test_embedder"),
        ("Full Pipeline", "tests.test_full_pipeline"),
    ]
    failed: list[str] = []
    for name, mod in modules:
        print(f"\n--- {name} ---")
        try:
            m = __import__(mod, fromlist=["run_tests"])
            m.run_tests()
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append(name)
    print("\n" + "=" * 50)
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    print("All tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
