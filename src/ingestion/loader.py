"""
PDF loader: read PDFs from a directory and return structured documents with metadata.
"""

from pathlib import Path
from dataclasses import dataclass
import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """
    A single document unit (e.g. one page) with text and metadata.

    Attributes:
        content: Extracted text content.
        metadata: Dict with at least "filename" and "page_number".
    """

    content: str
    metadata: dict


def _extract_text_safe(page) -> str:
    """
    Extract text from a single PDF page, returning a stripped string or empty string.

    Handles None and empty output from pypdf so callers don't need to.
    """
    try:
        text = page.extract_text()
    except Exception as e:
        logger.warning("Text extraction failed for a page: %s", e)
        return ""
    if text is None:
        return ""
    return text.strip()


def _load_single_pdf(path: Path) -> list[Document]:
    """
    Load one PDF file and return a list of Document (one per page with text).

    Skips pages that yield no text. Metadata includes filename and page_number.
    """
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
    """
    Load all PDFs from a directory and return a list of documents.

    Each PDF is processed page by page. Pages with no extractable text are skipped.
    Each returned Document has content (str) and metadata with "filename" and
    "page_number".

    Args:
        directory: Path to a folder containing PDF files.

    Returns:
        List of Document instances. Empty list if the directory does not exist,
        is not a directory, or contains no PDFs.

    Raises:
        Does not raise; PDFs that fail to open or read are logged and skipped.
    """
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
