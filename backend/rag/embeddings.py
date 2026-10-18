"""Optional dense embeddings.

Default backend is `fastembed` (ONNX, CPU, no PyTorch). It downloads the model
(`BAAI/bge-small-en-v1.5`, ~130 MB) from Hugging Face the first time it runs.
If fastembed is not installed or the model cannot be downloaded, the knowledge
base falls back to BM25 keyword retrieval and says so in /health.
"""
from __future__ import annotations

import os
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class FastEmbedEmbedder:
    def __init__(self, model_name: str | None = None):
        from fastembed import TextEmbedding  # imported lazily: optional dependency

        self.name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self._model = TextEmbedding(self.name)
        self.dim = len(self.embed_query("probe"))

    @staticmethod
    def _norm(m: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(n == 0, 1, n)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = np.array(list(self._model.embed(texts, batch_size=64)), dtype=np.float32)
        return self._norm(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        vec = np.array(list(self._model.query_embed([text]))[0], dtype=np.float32)
        return self._norm(vec)


def load_embedder(backend: str | None = None) -> Embedder | None:
    backend = (backend or os.getenv("EMBEDDINGS_BACKEND", "fastembed")).lower()
    if backend in ("none", "off", "bm25"):
        return None
    if backend == "fastembed":
        try:
            return FastEmbedEmbedder()
        except Exception as exc:  # not installed, no network for the model, etc.
            print(f"[rag] dense embeddings unavailable ({type(exc).__name__}: {exc}); using BM25 only")
            return None
    raise ValueError(f"unknown EMBEDDINGS_BACKEND '{backend}'")
