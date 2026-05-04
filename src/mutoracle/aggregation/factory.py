"""Aggregation strategy construction from config."""

from __future__ import annotations

from typing import Literal, Protocol

from mutoracle.aggregation.threshold import ConfidenceGatedAggregator
from mutoracle.aggregation.uniform import UniformAggregator
from mutoracle.aggregation.weighted import WeightedAggregator
from mutoracle.contracts import Aggregator


class AggregatorConfigLike(Protocol):
    strategy: Literal["uniform", "weighted", "confidence_gated"]
    weights: dict[str, float]
    confidence_gate_min_score: float
    confidence_gate_min_oracles: int


def build_aggregator(config: AggregatorConfigLike) -> Aggregator:
    """Return the configured aggregation strategy."""

    if config.strategy == "uniform":
        return UniformAggregator()
    if config.strategy == "confidence_gated":
        return ConfidenceGatedAggregator(
            weights=config.weights,
            min_score=float(config.confidence_gate_min_score),
            min_passing_oracles=config.confidence_gate_min_oracles,
        )
    return WeightedAggregator(config.weights)
