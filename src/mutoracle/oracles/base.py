"""Shared oracle interfaces and score helpers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Protocol

from mutoracle.cache import SQLiteCacheLedger, oracle_cache_key
from mutoracle.contracts import RAGRun


@dataclass(frozen=True)
class OracleScore:
    """Normalized oracle score plus audit metadata."""

    oracle_name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ScoringOracle(Protocol):
    """Oracle implementation with detailed and float-only scoring paths."""

    name: str

    def score(self, run: RAGRun) -> float:
        """Return a normalized score in the inclusive range [0, 1]."""

    def score_result(self, run: RAGRun) -> OracleScore:
        """Return a normalized score with metadata."""


class CacheBackedOracle:
    """Base class for oracles that cache expensive model work."""

    name: str
    model_name: str

    def __init__(self, *, ledger: SQLiteCacheLedger | None = None) -> None:
        self._ledger = ledger

    def score(self, run: RAGRun) -> float:
        """Return a normalized score in the inclusive range [0, 1]."""

        return self.score_result(run).value

    def score_result(self, run: RAGRun) -> OracleScore:
        """Return a normalized score with metadata."""

        payload = oracle_payload(run)
        input_hash = stable_hash(payload)
        cache_key = oracle_cache_key(
            oracle_name=self.name,
            model=self.model_name,
            payload={"input_hash": input_hash},
        )
        if self._ledger is not None:
            cached = self._ledger.lookup_oracle_score(cache_key)
            if cached is not None:
                metadata = dict(cached.metadata)
                metadata["cache_hit"] = True
                return OracleScore(
                    oracle_name=self.name,
                    value=clamp_score(cached.score),
                    metadata=metadata,
                )

        result = self._score_uncached(run, input_hash=input_hash)
        result = OracleScore(
            oracle_name=result.oracle_name,
            value=clamp_score(result.value),
            metadata={**result.metadata, "cache_hit": False},
        )
        if self._ledger is not None:
            self._ledger.store_oracle_score(
                cache_key=cache_key,
                oracle_name=self.name,
                input_hash=input_hash,
                score=result.value,
                metadata=result.metadata,
            )
        return result

    def _score_uncached(self, run: RAGRun, *, input_hash: str) -> OracleScore:
        raise NotImplementedError


def oracle_payload(run: RAGRun) -> dict[str, Any]:
    """Return the stable RAGRun fields that define oracle inputs."""

    return {
        "answer": run.answer,
        "passages": run.passages,
        "query": run.query,
    }


def context_text(run: RAGRun) -> str:
    """Return the premise/context text used by faithfulness oracles."""

    return "\n\n".join(passage.strip() for passage in run.passages if passage.strip())


def stable_hash(payload: dict[str, Any] | list[Any] | str) -> str:
    """Return a stable SHA-256 hash for JSON-compatible payloads."""

    if isinstance(payload, str):
        encoded = payload
    else:
        encoded = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def clamp_score(value: float) -> float:
    """Clamp finite numeric values to the inclusive range [0, 1]."""

    if not math.isfinite(value):
        return 0.0
    return min(1.0, max(0.0, float(value)))


def cosine_to_unit_interval(cosine: float) -> float:
    """Map cosine similarity from [-1, 1] into [0, 1]."""

    if not math.isfinite(cosine):
        return 0.0
    bounded = min(1.0, max(-1.0, cosine))
    return (bounded + 1.0) / 2.0


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two dense vectors."""

    if len(left) != len(right) or not left:
        msg = "Cosine similarity requires non-empty vectors with matching lengths."
        raise ValueError(msg)
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
