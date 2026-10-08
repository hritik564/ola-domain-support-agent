"""A shared, lazy local SentenceTransformers embedding service."""

from __future__ import annotations

import threading
from typing import Any

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_MODEL_CACHE: dict[str, Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _get_model(model_name: str) -> Any:
    with _MODEL_CACHE_LOCK:
        if model_name not in _MODEL_CACHE:
            from sentence_transformers import SentenceTransformer

            _MODEL_CACHE[model_name] = SentenceTransformer(model_name, device="cpu")
        return _MODEL_CACHE[model_name]


class LocalEmbeddingService:
    """Encode policy documents and queries with one cached local model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding texts must be non-empty strings.")
        if not texts:
            return []
        vectors = _get_model(self.model_name).encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(value) for value in vector] for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Query text must be a non-empty string.")
        return self.embed_documents([text])[0]
