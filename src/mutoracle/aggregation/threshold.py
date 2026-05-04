"""Confidence-gated oracle aggregation."""

from __future__ import annotations

from dataclasses import dataclass

from mutoracle.aggregation._scores import clamp_score
from mutoracle.aggregation.weighted import WeightedAggregator


@dataclass(frozen=True)
class ConfidenceGatedAggregator:
    """Weighted aggregation that gates low-confidence oracle sets to zero."""

    weights: dict[str, float]
    min_score: float = 0.5
    min_passing_oracles: int = 2

    name = "confidence_gated"

    def __post_init__(self) -> None:
        WeightedAggregator(self.weights)
        if self.min_passing_oracles < 1:
            msg = "min_passing_oracles must be at least 1."
            raise ValueError(msg)

    def combine(self, scores: dict[str, float]) -> float:
        """Return weighted score only when enough oracle scores pass the gate."""

        gate = clamp_score(float(self.min_score))
        passing = sum(
            1
            for oracle_name in self.weights
            if clamp_score(float(scores.get(oracle_name, 0.0))) >= gate
        )
        if passing < self.min_passing_oracles:
            return 0.0
        return WeightedAggregator(self.weights).combine(scores)
