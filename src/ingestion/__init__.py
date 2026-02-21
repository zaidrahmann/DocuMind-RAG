"""
Document ingestion: load files (e.g. PDFs) and convert them to a uniform document format.
"""

from .loader import load_pdfs_from_directory, Document

__all__ = ["load_pdfs_from_directory", "Document"]
