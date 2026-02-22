"""Tests for the PDF loader."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion import load_pdfs_from_directory, Document

# Load create_sample_pdf by path (avoids package import issues when run as script)
_spec = importlib.util.spec_from_file_location(
    "create_sample_pdf", ROOT / "tests" / "sample_data" / "create_sample_pdf.py"
)
_create_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_create_mod)  # type: ignore[union-attr]
create_sample_pdf = _create_mod.create_sample_pdf


def test_load_empty_directory() -> None:
    """Empty directory should return empty list."""
    d = Path(__file__).parent / "_tmp_empty"
    d.mkdir(exist_ok=True)
    try:
        docs = load_pdfs_from_directory(d)
        assert docs == []
    finally:
        if d.exists() and not any(d.iterdir()):
            d.rmdir()


def test_load_nonexistent_directory() -> None:
    """Non-existent path should return empty list (loader logs and returns [])."""
    docs = load_pdfs_from_directory(Path("/nonexistent/path/12345"))
    assert docs == []


def test_load_directory_with_blank_pdf() -> None:
    """Directory with a blank PDF (no extractable text) returns empty list - pages are skipped."""
    d = Path(__file__).parent / "_tmp_blank"
    d.mkdir(exist_ok=True)
    try:
        create_sample_pdf(d / "blank.pdf")
        docs = load_pdfs_from_directory(d)
        assert docs == []
    finally:
        for f in d.glob("*"):
            f.unlink()
        d.rmdir()


def test_load_handles_pdf_structure() -> None:
    """Loader should not crash on valid PDF; returns list of Document (possibly empty)."""
    d = Path(__file__).parent / "_tmp_pdf"
    d.mkdir(exist_ok=True)
    try:
        create_sample_pdf(d / "a.pdf")
        docs = load_pdfs_from_directory(d)
        assert isinstance(docs, list)
        assert len(docs) == 0
    finally:
        for f in d.glob("*"):
            f.unlink()
        d.rmdir()


def run_tests() -> None:
    """Run all loader tests."""
    tests = [
        test_load_empty_directory,
        test_load_nonexistent_directory,
        test_load_directory_with_blank_pdf,
        test_load_handles_pdf_structure,
    ]
    for t in tests:
        t()
        print(f"  OK {t.__name__}")
    print(f"\nAll {len(tests)} loader tests passed.")
    print("Tip: Add a real PDF with text to tests/sample_data/ to verify full extraction.")


if __name__ == "__main__":
    run_tests()
