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
        return self._score_uncached_many([run], input_hashes=[input_hash])[0]

    def _score_uncached_many(
        self,
        runs: Sequence[RAGRun],
        *,
        input_hashes: Sequence[str],
    ) -> list[OracleScore]:
        prepared: list[tuple[str, str, str] | None] = []
        texts: list[str] = []
        for run, input_hash in zip(runs, input_hashes, strict=True):
            context = context_text(run)
            answer = run.answer.strip()
            if not context or not answer:
                prepared.append(None)
                continue
            prepared.append((input_hash, context, answer))
            texts.extend([context, answer])

        vectors = list(self._backend.encode(texts)) if texts else []
        vector_index = 0
        results: list[OracleScore] = []
        for item, input_hash in zip(prepared, input_hashes, strict=True):
            if item is None:
                results.append(
                    OracleScore(
                        oracle_name=self.name,
                        value=0.0,
                        metadata={
                            "input_hash": input_hash,
                            "model": self.model_name,
                            "reason": "empty_context_or_answer",
                        },
                    )
                )
                continue
            context_vector = vectors[vector_index]
            answer_vector = vectors[vector_index + 1]
            vector_index += 2
            raw_cosine = cosine_similarity(list(context_vector), list(answer_vector))
            score = cosine_to_unit_interval(raw_cosine)
            results.append(
                OracleScore(
                    oracle_name=self.name,
                    value=score,
                    metadata={
                        "input_hash": item[0],
                        "model": self.model_name,
                        "raw_cosine": raw_cosine,
                    },
                )
            )
        return results
