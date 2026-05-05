"""Uniform oracle-score aggregation."""

from __future__ import annotations

from collections.abc import Mapping

from mutoracle.aggregation._scores import clamp_score


class UniformAggregator:
    """Combine available oracle scores with equal weight."""

    name = "uniform"

    def combine(self, scores: dict[str, float]) -> float:
        """Return the arithmetic mean of valid normalized oracle scores."""

        values = [_normalized(value) for value in scores.values()]
        if not values:
            return 0.0
        return clamp_score(sum(values) / len(values))


def uniform_score(scores: Mapping[str, float]) -> float:
    """Convenience helper for one-off uniform aggregation."""

    return UniformAggregator().combine(dict(scores))


def _normalized(value: float) -> float:
    return clamp_score(float(value))
