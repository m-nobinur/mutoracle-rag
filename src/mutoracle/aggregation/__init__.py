"""Aggregation strategy exports."""

from mutoracle.aggregation.threshold import ConfidenceGatedAggregator
from mutoracle.aggregation.uniform import UniformAggregator, uniform_score
from mutoracle.aggregation.weighted import WeightedAggregator, validate_weights

__all__ = [
    "ConfidenceGatedAggregator",
    "UniformAggregator",
    "WeightedAggregator",
    "build_aggregator",
    "uniform_score",
    "validate_weights",
]

from mutoracle.aggregation.factory import build_aggregator
