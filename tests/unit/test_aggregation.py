from __future__ import annotations

import pytest

from mutoracle.aggregation import (
    ConfidenceGatedAggregator,
    UniformAggregator,
    WeightedAggregator,
)


def test_uniform_aggregator_averages_available_scores() -> None:
    aggregator = UniformAggregator()

    assert aggregator.combine({"nli": 0.2, "semantic_similarity": 0.8}) == 0.5


def test_weighted_aggregator_requires_weights_to_sum_to_one() -> None:
    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        WeightedAggregator({"nli": 0.6, "semantic_similarity": 0.6})


def test_weighted_aggregator_treats_missing_scores_as_zero() -> None:
    aggregator = WeightedAggregator({"nli": 0.7, "semantic_similarity": 0.3})

    assert aggregator.combine({"nli": 1.0}) == pytest.approx(0.7)


def test_confidence_gated_aggregator_blocks_low_confidence_score_sets() -> None:
    aggregator = ConfidenceGatedAggregator(
        {"nli": 0.5, "semantic_similarity": 0.5},
        min_score=0.6,
        min_passing_oracles=2,
    )

    assert aggregator.combine({"nli": 0.8, "semantic_similarity": 0.4}) == 0.0
    assert aggregator.combine({"nli": 0.8, "semantic_similarity": 0.6}) == 0.7


def test_confidence_gated_rejects_impossible_gate_requirement() -> None:
    with pytest.raises(ValueError, match=r"cannot exceed"):
        ConfidenceGatedAggregator({"nli": 1.0}, min_passing_oracles=2)
