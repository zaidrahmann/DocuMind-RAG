"""FAISS vector store for production-grade RAG systems.

Uses IndexFlatIP (inner product) with L2-normalized vectors to approximate
cosine similarity. Metadata is stored separately in a list, indexed by FAISS ids.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def _normalize_vectors(x: np.ndarray) -> np.ndarray:
    """L2-normalize vectors in-place. Returns the array for chaining."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    x /= norms
    return x


def _ensure_float32(x: np.ndarray) -> np.ndarray:
    """Ensure array is contiguous float32."""
    out = np.ascontiguousarray(x, dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    return out


class FaissVectorStore:
    """FAISS vector store using IndexFlatIP for cosine similarity search.

    Embeddings must be L2-normalized before adding/querying so that
    inner product equals cosine similarity.
    """

    def __init__(self, embedding_dim: int) -> None:
        """Initialize the vector store with a fixed embedding dimension.

        Args:
            embedding_dim: Dimension of embedding vectors (must match your embedder).
        """
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")
        self.embedding_dim = int(embedding_dim)
        self._index: faiss.IndexFlatIP | None = None
        self._metadata: list[dict] = []
        self._built = False

    def build_index(self) -> None:
        """Create the FAISS index. Call before add_embeddings if starting fresh."""
        self._index = faiss.IndexFlatIP(self.embedding_dim)
        self._metadata = []
        self._built = True
        logger.debug("Built FAISS IndexFlatIP with dim=%d", self.embedding_dim)

    def add_embeddings(self, embeddings: np.ndarray, metadata: list[dict]) -> None:
        """Add embeddings and corresponding metadata to the index.

        Embeddings are normalized and converted to float32. Order of metadata
        must match the order of embeddings (one dict per row).

        Args:
            embeddings: (N, embedding_dim) array; will be L2-normalized.
            metadata: List of N dicts (JSON-serializable for save/load).
        """
        if self._index is None:
            self.build_index()

        embeddings = _ensure_float32(embeddings)
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dim {embeddings.shape[1]} != store dim {self.embedding_dim}"
            )
        if len(metadata) != embeddings.shape[0]:
            raise ValueError(
                f"metadata length ({len(metadata)}) must equal embeddings rows ({embeddings.shape[0]})"
            )

        _normalize_vectors(embeddings)
        self._index.add(embeddings)
        self._metadata.extend(metadata)
        logger.debug("Added %d vectors; total=%d", embeddings.shape[0], self._index.ntotal)

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[float, dict]]:
        """Search for the k nearest neighbors by cosine similarity (inner product on normalized vectors).

        Args:
            query_vector: 1D or 2D array of shape (embedding_dim,) or (1, embedding_dim).
            k: Number of results to return.

        Returns:
            List of (score, metadata) tuples, ordered by descending similarity.
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        q = _ensure_float32(query_vector)
        if q.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Query dim {q.shape[1]} != store dim {self.embedding_dim}"
            )
        _normalize_vectors(q)

        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results: list[tuple[float, dict]] = []
        for s, i in zip(scores[0], indices[0]):
            if i < 0:
                continue
            results.append((float(s), self._metadata[i]))
        return results

    def save(self, path: str | Path) -> None:
        """Persist the index and metadata to disk.

        Saves:
            - {path}.index: FAISS binary index
            - {path}.meta.json: Metadata list (JSON)
        """
        if self._index is None:
            raise RuntimeError("No index to save; call build_index and add_embeddings first")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        index_path = path if path.suffix == ".index" else path.with_suffix(".index")
        meta_path = index_path.with_suffix(".meta.json")

        faiss.write_index(self._index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=0)
        logger.debug("Saved index to %s and metadata to %s", index_path, meta_path)

    def load(self, path: str | Path) -> None:
        """Load index and metadata from disk.

        Expects:
            - {path}.index (or path ending in .index)
            - {path}.meta.json
        """
        path = Path(path)
        index_path = path if path.suffix == ".index" else path.with_suffix(".index")
        meta_path = index_path.with_suffix(".meta.json")

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        self._index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        if self._index.d != self.embedding_dim:
            raise ValueError(
                f"Loaded index dim {self._index.d} != store embedding_dim {self.embedding_dim}"
            )
        if len(self._metadata) != self._index.ntotal:
            raise ValueError(
                f"Metadata count ({len(self._metadata)}) != index size ({self._index.ntotal})"
            )

        self._built = True
        logger.debug("Loaded index with %d vectors from %s", self._index.ntotal, index_path)
