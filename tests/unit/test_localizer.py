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


@dataclass(frozen=True)
class QueryOnlyPromptMutation:
    operator_id: str = "QN"
    stage: Stage = "prompt"
    name: str = "Query Shift"

    def apply(self, run: RAGRun, *, rng: Random) -> RAGRun:
        del rng
        return clone_run(
            run,
            query=f"{run.query} not",
            operator_id=self.operator_id,
            operator_name=self.name,
            stage=self.stage,
        )


@dataclass(frozen=True)
class RandomScoreMutation:
    operator_id: str
    stage: Stage
    name: str = "Random score mutation"

    def apply(self, run: RAGRun, *, rng: Random) -> RAGRun:
        mutated = clone_run(
            run,
            operator_id=self.operator_id,
            operator_name=self.name,
            stage=self.stage,
        )
        metadata = dict(mutated.metadata)
        metadata["fixture_score"] = rng.random()
        return RAGRun(
            query=mutated.query,
            passages=mutated.passages,
            answer=mutated.answer,
            metadata=metadata,
        )


class QueryDependentPipeline:
    def run(self, query: str) -> RAGRun:
        answer = "unsupported answer" if "not" in query.lower() else "supported answer"
        return RAGRun(
            query=query,
            passages=["supported context"],
            answer=answer,
            metadata={},
        )


class AnswerSupportOracle:
    name = "nli"

    def score(self, run: RAGRun) -> float:
        return 0.2 if run.answer.startswith("unsupported") else 1.0


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


def test_prompt_mutations_are_scored_from_rerun_pipeline_outputs() -> None:
    localizer = FaultLocalizer(
        pipeline=QueryDependentPipeline(),
        oracles=[AnswerSupportOracle()],
        aggregator=UniformAggregator(),
        delta_threshold=0.1,
        operators={"QN": QueryOnlyPromptMutation()},
    )

    report = localizer.diagnose("What is supported?")

    assert report.stage == "prompt"
    assert report.deltas["QN"] == pytest.approx(0.8)


def test_explicit_empty_operator_mapping_is_respected() -> None:
    localizer = FaultLocalizer(
        pipeline=StaticPipeline(),
        oracles=[MetadataOracle()],
        aggregator=UniformAggregator(),
        delta_threshold=0.05,
        operators={},
    )

    report = localizer.diagnose("What is supported?")

    assert report.deltas == {}
    assert report.stage_deltas == {"retrieval": 0.0, "prompt": 0.0, "generation": 0.0}
    assert report.stage == "no_fault_detected"


def test_operator_order_does_not_change_randomized_mutation_scores() -> None:
    forward = FaultLocalizer(
        pipeline=StaticPipeline(),
        oracles=[MetadataOracle()],
        aggregator=UniformAggregator(),
        delta_threshold=0.05,
        operators={
            "CI": RandomScoreMutation("CI", "retrieval"),
            "QN": RandomScoreMutation("QN", "prompt"),
        },
        seed=2026,
    )
    reversed_order = FaultLocalizer(
        pipeline=StaticPipeline(),
        oracles=[MetadataOracle()],
        aggregator=UniformAggregator(),
        delta_threshold=0.05,
        operators={
            "QN": RandomScoreMutation("QN", "prompt"),
            "CI": RandomScoreMutation("CI", "retrieval"),
        },
        seed=2026,
    )

    report_forward = forward.diagnose("What is supported?")
    report_reversed = reversed_order.diagnose("What is supported?")

    assert report_forward.deltas == report_reversed.deltas
    assert report_forward.stage_deltas == report_reversed.stage_deltas


def test_localizer_rejects_delta_threshold_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"delta_threshold"):
        FaultLocalizer(
            pipeline=StaticPipeline(),
            oracles=[MetadataOracle()],
            aggregator=UniformAggregator(),
            delta_threshold=1.1,
            operators={"CI": ScoreMutation("CI", "retrieval", 0.9)},
        )
