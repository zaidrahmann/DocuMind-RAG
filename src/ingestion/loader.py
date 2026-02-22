"""Load PDFs from a directory into Document objects."""

from pathlib import Path
from dataclasses import dataclass
import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document unit with content and metadata (filename, page_number, etc.)."""

    content: str
    metadata: dict


def _extract_text_safe(page) -> str:
    """Extract text from a PDF page; returns stripped string or empty string."""
    try:
        text = page.extract_text()
    except Exception as e:
        logger.warning("Text extraction failed for a page: %s", e)
        return ""
    if text is None:
        return ""
    return text.strip()


def _load_single_pdf(path: Path) -> list[Document]:
    """Load one PDF; returns Document per page (skips empty pages)."""
    documents: list[Document] = []
    filename = path.name

    try:
        reader = PdfReader(str(path))
    except Exception as e:
        logger.error("Failed to open PDF %s: %s", path, e)
        raise

    for page_number, page in enumerate(reader.pages, start=1):
        text = _extract_text_safe(page)
        if not text:
            continue
        documents.append(
            Document(
                content=text,
                metadata={"filename": filename, "page_number": page_number},
            )
        )

    return documents


def load_pdfs_from_directory(directory: str | Path) -> list[Document]:
    """Load all PDFs from a directory; returns Document per page. Logs and skips failures."""
    directory = Path(directory)
    if not directory.is_dir():
        logger.warning("Not a directory or does not exist: %s", directory)
        return []

    all_documents: list[Document] = []
    pdf_paths = sorted(directory.glob("*.pdf"))

    for path in pdf_paths:
        try:
            docs = _load_single_pdf(path)
            all_documents.extend(docs)
        except Exception as e:
            logger.exception("Skipping PDF %s due to error: %s", path, e)
            # Skip this file and continue with others
            continue

    return all_documents
