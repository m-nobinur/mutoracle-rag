"""FAISS index adapter for embedding-backed retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

from mutoracle.retrieval import Passage, RetrievalHit


class Embedder(Protocol):
    """Minimal interface shared by sentence-transformers style embedders."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Return one dense vector per text."""


@dataclass(frozen=True)
class IndexedPassage:
    """Passage and vector pair stored in an index."""

    passage: Passage
    vector: list[float]


class FaissIndex:
    """Small nearest-neighbor wrapper with native FAISS when available.

    The fixture smoke path does not require FAISS. If `faiss` and `numpy` are
    installed, this class uses `IndexFlatIP`; otherwise it falls back to the same
    cosine ordering in pure Python so tests remain lightweight.
    """

    def __init__(self, passages: list[Passage], embedder: Embedder) -> None:
        if not passages:
            msg = "FAISS index requires at least one passage."
            raise ValueError(msg)
        self._passages = passages
        self._embedder = embedder
        self._vectors = self._embedder.encode(
            [f"{passage.title} {passage.text}" for passage in passages]
        )
        self._validate_vectors(self._vectors)
        self._native_index: tuple[Any, Any] | None = _build_native_index(self._vectors)

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        """Return top-k passages by cosine similarity."""

        if top_k < 1:
            msg = "top_k must be at least 1."
            raise ValueError(msg)
        query_vectors = self._embedder.encode([query])
        self._validate_vectors(query_vectors)
        query_vector = query_vectors[0]
        if self._native_index is not None:
            return self._native_search(query_vector, top_k=top_k)
        scored = [
            RetrievalHit(passage=passage, score=round(_cosine(query_vector, vector), 6))
            for passage, vector in zip(self._passages, self._vectors, strict=True)
        ]
        scored.sort(key=lambda hit: (-hit.score, hit.passage.id))
        return scored[:top_k]

    def _native_search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
    ) -> list[RetrievalHit]:
        if self._native_index is None:
            msg = "Native FAISS index is unavailable."
            raise RuntimeError(msg)
        faiss_index, np_module = self._native_index
        query_array = np_module.asarray([query_vector], dtype="float32")
        faiss_module = import_module("faiss")
        faiss_module.normalize_L2(query_array)
        scores, indices = faiss_index.search(query_array, top_k)
        hits = [
            RetrievalHit(
                passage=self._passages[int(index)],
                score=round(float(score), 6),
            )
            for score, index in zip(scores[0], indices[0], strict=True)
            if int(index) >= 0
        ]
        hits.sort(key=lambda hit: (-hit.score, hit.passage.id))
        return hits

    def _validate_vectors(self, vectors: list[list[float]]) -> None:
        if not vectors or not vectors[0]:
            msg = "Embedding vectors must be non-empty."
            raise ValueError(msg)
        dimension = len(vectors[0])
        if any(len(vector) != dimension for vector in vectors):
            msg = "Embedding vectors must have a consistent dimension."
            raise ValueError(msg)


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )
    left_norm = sum(value * value for value in left) ** 0.5
    right_norm = sum(value * value for value in right) ** 0.5
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(dot / (left_norm * right_norm))


def _build_native_index(vectors: list[list[float]]) -> tuple[Any, Any] | None:
    try:
        faiss_module = import_module("faiss")
        np_module = import_module("numpy")
    except ModuleNotFoundError:
        return None

    vector_array = np_module.asarray(vectors, dtype="float32")
    faiss_module.normalize_L2(vector_array)
    index = faiss_module.IndexFlatIP(vector_array.shape[1])
    index.add(vector_array)
    return index, np_module
