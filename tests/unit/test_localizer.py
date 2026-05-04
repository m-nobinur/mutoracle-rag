from __future__ import annotations

from dataclasses import dataclass
from random import Random

import pytest

from mutoracle.aggregation import UniformAggregator
from mutoracle.config import AggregationConfig
from mutoracle.contracts import RAGRun, Stage
from mutoracle.localizer import FaultLocalizer, compute_stage_deltas
from mutoracle.mutations.base import clone_run


class StaticPipeline:
    def run(self, query: str) -> RAGRun:
        return RAGRun(
            query=query,
            passages=["supported context"],
            answer="supported answer",
            metadata={"fixture_score": 1.0},
        )


@dataclass(frozen=True)
class ScoreMutation:
    operator_id: str
    stage: Stage
    score: float
    name: str = "Score Mutation"

    def apply(self, run: RAGRun, *, rng: Random) -> RAGRun:
        del rng
        mutated = clone_run(
            run,
            operator_id=self.operator_id,
            operator_name=self.name,
            stage=self.stage,
        )
        metadata = dict(mutated.metadata)
        metadata["fixture_score"] = self.score
        return RAGRun(
            query=mutated.query,
            passages=mutated.passages,
            answer=mutated.answer,
            metadata=metadata,
        )


class MetadataOracle:
    name = "fixture"

    def score(self, run: RAGRun) -> float:
        return float(run.metadata["fixture_score"])


def test_worked_example_attributes_largest_delta_stage() -> None:
    localizer = FaultLocalizer(
        pipeline=StaticPipeline(),
        oracles=[MetadataOracle()],
        aggregator=UniformAggregator(),
        delta_threshold=0.05,
        operators={
            "CI": ScoreMutation("CI", "retrieval", 0.4),
            "QN": ScoreMutation("QN", "prompt", 0.9),
            "FA": ScoreMutation("FA", "generation", 0.8),
        },
    )

    report = localizer.diagnose("What is supported?")

    assert report.stage == "retrieval"
    assert report.confidence == pytest.approx(0.6 / (0.6 + 0.1 + 0.2))
    assert report.deltas["CI"] == pytest.approx(0.6)
    assert report.stage_deltas["retrieval"] == pytest.approx(0.6)


def test_threshold_controls_no_fault_decision() -> None:
    aggregation = AggregationConfig(delta_threshold=0.2)
    localizer = FaultLocalizer(
        pipeline=StaticPipeline(),
        oracles=[MetadataOracle()],
        aggregator=UniformAggregator(),
        delta_threshold=float(aggregation.delta_threshold),
        operators={"CI": ScoreMutation("CI", "retrieval", 0.9)},
    )

    report = localizer.diagnose("What is supported?")

    assert report.stage == "no_fault_detected"
    assert report.confidence == 0.0


def test_negative_and_missing_deltas_do_not_create_confidence() -> None:
    stage_deltas = compute_stage_deltas(
        {"CI": -0.2},
        {"CI": ScoreMutation("CI", "retrieval", 1.2)},
    )
    localizer = FaultLocalizer(
        pipeline=StaticPipeline(),
        oracles=[MetadataOracle()],
        aggregator=UniformAggregator(),
        delta_threshold=0.05,
        operators={"CI": ScoreMutation("CI", "retrieval", 1.2)},
    )

    report = localizer.diagnose("What is supported?")

    assert stage_deltas == {"retrieval": -0.2, "prompt": 0.0, "generation": 0.0}
    assert report.stage == "no_fault_detected"
    assert report.confidence == 0.0
