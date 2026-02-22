"""Multilingual sentence-transformer embedding wrapper (singleton, batched)."""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]


class MultilingualEmbedder:
    """Multilingual SentenceTransformer wrapper; singleton model instance."""

    _instance: MultilingualEmbedder | None = None
    _model: SentenceTransformer | None = None

    def __new__(cls, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", **kwargs) -> MultilingualEmbedder:
        if cls._instance is None:
            obj = super().__new__(cls)
            cls._instance = obj
        return cls._instance

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        device: str | None = None,
        **model_kwargs,
    ) -> None:
        if MultilingualEmbedder._model is None:
            MultilingualEmbedder._model = SentenceTransformer(model_name, device=device, **model_kwargs)

    @property
    def model(self) -> SentenceTransformer:
        if MultilingualEmbedder._model is None:
            raise RuntimeError("Embedder not initialized: call MultilingualEmbedder(model_name=...) first.")
        return MultilingualEmbedder._model

    def encode_documents(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize_embeddings: bool = False,
        **encode_kwargs,
    ) -> np.ndarray:
        """Encode document strings into an (N, dim) embedding matrix."""
        if not texts:
            return np.zeros((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            **encode_kwargs,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(
        self,
        text: str,
        normalize_embeddings: bool = False,
        **encode_kwargs,
    ) -> np.ndarray:
        """Encode a single query into a 1D embedding vector."""
        embedding = self.model.encode(
            [text],
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            **encode_kwargs,
        )
        return np.asarray(embedding[0], dtype=np.float32)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton and unload model."""
        cls._instance = None
        cls._model = None
