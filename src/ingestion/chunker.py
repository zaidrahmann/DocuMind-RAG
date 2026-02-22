from dataclasses import dataclass
import logging
from typing import Any

logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """Text chunk with inherited document metadata plus chunk_index and token_count."""

    content: str
    metadata: dict[str, Any]


def chunk_documents(
    documents: list[Any],
    *,
    chunk_size: int = 512,
    overlap: int = 64,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Split documents into token-based overlapping chunks. Requires tiktoken."""
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size")

    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)
    except ImportError as e:
        raise ImportError(
            "Token-based chunking requires tiktoken. Install with: pip install tiktoken"
        ) from e

    step = chunk_size - overlap
    chunks: list[Chunk] = []

    for doc in documents:
        content = getattr(doc, "content", None)
        meta = getattr(doc, "metadata", None)
        if content is None or meta is None:
            logger.warning("Document missing .content or .metadata, skipping: %s", type(doc))
            continue
        if not isinstance(meta, dict):
            meta = dict(meta) if meta else {}
        text = content if isinstance(content, str) else str(content)
        if not text.strip():
            continue

        tokens = enc.encode(text)
        if not tokens:
            continue

        base_meta = dict(meta)
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            window = tokens[start:end]
            chunk_text = enc.decode(window)

            chunk_meta = {**base_meta, "chunk_index": chunk_index, "token_count": len(window)}
            chunks.append(Chunk(content=chunk_text, metadata=chunk_meta))

            if end >= len(tokens):
                break
            start += step
            chunk_index += 1

    return chunks
