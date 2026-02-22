"""Create a minimal test PDF using pypdf (no extra deps)."""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfWriter


def create_sample_pdf(out_path: Path | None = None) -> Path:
    """Write a minimal test PDF with one page. Blank pages have no extractable text;
    use for testing loader machinery. Place a real PDF in sample_data/ for full tests.
    """
    if out_path is None:
        out_path = Path(__file__).parent / "test_document.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with open(out_path, "wb") as f:
        writer.write(f)
    return out_path


if __name__ == "__main__":
    p = create_sample_pdf()
    print(f"Created: {p}")
