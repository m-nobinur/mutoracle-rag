"""Weighted oracle-score aggregation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from mutoracle.aggregation._scores import clamp_score

WEIGHT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class WeightedAggregator:
    """Combine oracle scores using a validated convex weight vector."""

    weights: dict[str, float]

    name = "weighted"

    def __post_init__(self) -> None:
        validate_weights(self.weights)

    def combine(self, scores: dict[str, float]) -> float:
        """Return the weighted sum, treating absent oracle scores as zero."""

        total = 0.0
        for oracle_name, weight in self.weights.items():
            total += weight * clamp_score(float(scores.get(oracle_name, 0.0)))
        return clamp_score(total)


def validate_weights(weights: Mapping[str, float]) -> None:
    """Validate that oracle weights are finite, non-negative, and sum to one."""

    if not weights:
        msg = "Aggregator weights must not be empty."
        raise ValueError(msg)
    total = 0.0
    for oracle_name, weight in weights.items():
        if not oracle_name:
            msg = "Aggregator weight names must be non-empty."
            raise ValueError(msg)
        if not math.isfinite(weight) or weight < 0.0:
            msg = f"Aggregator weight for {oracle_name!r} must be finite and >= 0."
            raise ValueError(msg)
        total += float(weight)
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        msg = f"Aggregator weights must sum to 1.0; got {total:.6f}."
        raise ValueError(msg)
