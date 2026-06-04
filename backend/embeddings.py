"""
Local text embeddings via fastembed (ONNX) — no PyTorch.

We deliberately use fastembed instead of sentence-transformers: it runs the
same `BAAI/bge-small-en-v1.5` model through ONNX Runtime, producing 384-dim
L2-normalized vectors, but without pulling in the ~1 GB PyTorch dependency.
That keeps the deploy image small and the cold start fast.

One shared, cached model instance is reused across ingestion, the API, and eval.
"""

from __future__ import annotations

from functools import lru_cache

from backend.config import get_settings


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=get_settings().embedding_model)


def load() -> None:
    """Force the model to initialize now (e.g. at server startup)."""
    _model()


def embed_one(text: str) -> list[float]:
    """Embed a single string → 384-dim list[float] (L2-normalized)."""
    return next(iter(_model().embed([text]))).tolist()


def embed_many(texts: list[str], batch_size: int = 16) -> list[list[float]]:
    """Embed many strings. A small batch keeps ONNX Runtime memory bounded
    (large batches blow up the attention tensor on CPU)."""
    return [v.tolist() for v in _model().embed(texts, batch_size=batch_size)]
