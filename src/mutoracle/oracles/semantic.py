"""Semantic-similarity oracle backed by sentence-transformers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from mutoracle.cache import SQLiteCacheLedger
from mutoracle.config import MutOracleConfig
from mutoracle.contracts import RAGRun
from mutoracle.oracles.base import (
    CacheBackedOracle,
    OracleScore,
    context_text,
    cosine_similarity,
    cosine_to_unit_interval,
)


class EmbeddingBackend(Protocol):
    """Minimal embedding backend used by the semantic oracle."""

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return one dense vector per input text."""


class SentenceTransformerBackend:
    """Lazy sentence-transformers adapter."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any | None = None

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """Return sentence-transformer embeddings."""

        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                msg = (
                    "sentence-transformers is required for SemanticSimilarityOracle. "
                    "Install the oracle extras or inject an EmbeddingBackend."
                )
                raise RuntimeError(msg) from error
            self._model = SentenceTransformer(self.model_name)
        vectors = self._model.encode(list(texts), normalize_embeddings=False)
        return [[float(value) for value in vector] for vector in vectors]


class SemanticSimilarityOracle(CacheBackedOracle):
    """Scores answer/context semantic relatedness with normalized cosine."""

    name = "semantic_similarity"

    def __init__(
        self,
        *,
        config: MutOracleConfig | None = None,
        ledger: SQLiteCacheLedger | None = None,
        backend: EmbeddingBackend | None = None,
        model_name: str | None = None,
    ) -> None:
        super().__init__(ledger=ledger)
        self.model_name = (
            model_name
            or (config.oracles.semantic_model if config is not None else None)
            or "sentence-transformers/all-mpnet-base-v2"
        )
        self._backend = backend or SentenceTransformerBackend(self.model_name)

    def _score_uncached(self, run: RAGRun, *, input_hash: str) -> OracleScore:
        context = context_text(run)
        if not context or not run.answer.strip():
            return OracleScore(
                oracle_name=self.name,
                value=0.0,
                metadata={
                    "input_hash": input_hash,
                    "model": self.model_name,
                    "reason": "empty_context_or_answer",
                },
            )

        context_vector, answer_vector = self._backend.encode([context, run.answer])
        raw_cosine = cosine_similarity(list(context_vector), list(answer_vector))
        score = cosine_to_unit_interval(raw_cosine)
        return OracleScore(
            oracle_name=self.name,
            value=score,
            metadata={
                "input_hash": input_hash,
                "model": self.model_name,
                "raw_cosine": raw_cosine,
            },
        )
