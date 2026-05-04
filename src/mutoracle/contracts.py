"""Public interfaces shared across MutOracle-RAG modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Literal, Protocol

Stage = Literal["retrieval", "prompt", "generation"]
DiagnosisStage = Literal["retrieval", "prompt", "generation", "no_fault_detected"]


@dataclass(frozen=True)
class RAGRun:
    """Observed output and intermediate artifacts from a RAG system."""

    query: str
    passages: list[str]
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGPipeline(Protocol):
    """Black-box system under test."""

    def run(self, query: str) -> RAGRun:
        """Return a baseline RAG run for a query."""


class MutationOperator(Protocol):
    """A controlled perturbation targeting one pipeline stage."""

    name: str
    stage: Stage

    def apply(self, run: RAGRun, *, rng: Random) -> RAGRun:
        """Return a mutated copy of a baseline run."""


class Oracle(Protocol):
    """Scores a RAG run for consistency or faithfulness."""

    name: str

    def score(self, run: RAGRun) -> float:
        """Return a normalized score in the inclusive range [0, 1]."""


class Aggregator(Protocol):
    """Combines oracle scores into one composite score."""

    def combine(self, scores: dict[str, float]) -> float:
        """Return the aggregate oracle score."""


@dataclass(frozen=True)
class FaultReport:
    """Fault-localization result for a single query."""

    stage: DiagnosisStage
    confidence: float
    deltas: dict[str, float]
    stage_deltas: dict[Stage, float]
    evidence: list[str] = field(default_factory=list)
