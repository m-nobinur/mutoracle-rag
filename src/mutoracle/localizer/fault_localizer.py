"""Transparent mutation-delta fault localization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from random import Random
from typing import Any, Protocol

from mutoracle.contracts import (
    Aggregator,
    DiagnosisStage,
    FaultReport,
    MutationOperator,
    RAGPipeline,
    RAGRun,
    Stage,
)
from mutoracle.mutations import mutation_registry
from mutoracle.oracles import clamp_score

STAGES: tuple[Stage, ...] = ("retrieval", "prompt", "generation")


class ScoreOracle(Protocol):
    """Oracle interface accepted by the localizer."""

    name: str

    def score(self, run: RAGRun) -> float:
        """Return a normalized score."""


class FaultLocalizer:
    """Diagnose the most likely faulty RAG stage from mutation score deltas."""

    def __init__(
        self,
        *,
        pipeline: RAGPipeline,
        oracles: Sequence[ScoreOracle],
        aggregator: Aggregator,
        delta_threshold: float,
        operators: Mapping[str, MutationOperator] | None = None,
        seed: int = 2026,
    ) -> None:
        self._pipeline = pipeline
        self._oracles = list(oracles)
        if not self._oracles:
            msg = "FaultLocalizer requires at least one oracle."
            raise ValueError(msg)
        self._aggregator = aggregator
        self._delta_threshold = clamp_score(float(delta_threshold))
        self._operators = dict(operators or mutation_registry())
        self._seed = seed

    def diagnose(self, query: str) -> FaultReport:
        """Return a stage-level fault report for one query."""

        baseline = self._pipeline.run(query)
        baseline_scores = score_run(baseline, self._oracles)
        baseline_omega = self._aggregator.combine(baseline_scores)
        rng = Random(self._seed)

        deltas: dict[str, float] = {}
        evidence = [
            f"Baseline composite score: {baseline_omega:.4f}.",
            f"Delta threshold: {self._delta_threshold:.4f}.",
        ]

        for operator_id, operator in self._operators.items():
            mutated = operator.apply(baseline, rng=rng)
            mutation = mutated.metadata.get("mutation", {})
            if isinstance(mutation, dict) and mutation.get("rejected") is True:
                deltas[operator_id] = 0.0
                reason = mutation.get("rejection_reason", "mutation rejected")
                evidence.append(f"{operator_id} skipped: {reason}.")
                continue

            mutated_scores = score_run(mutated, self._oracles)
            mutated_omega = self._aggregator.combine(mutated_scores)
            delta = baseline_omega - mutated_omega
            deltas[operator_id] = delta
            evidence.append(
                f"{operator_id} ({operator.stage}) delta: {delta:.4f} "
                f"from mutated score {mutated_omega:.4f}."
            )

        stage_deltas = compute_stage_deltas(deltas, self._operators)
        stage = choose_stage(stage_deltas, delta_threshold=self._delta_threshold)
        confidence = confidence_for_stage(stage, stage_deltas)
        evidence.append(f"Predicted stage: {stage} with confidence {confidence:.4f}.")
        return FaultReport(
            stage=stage,
            confidence=confidence,
            deltas=deltas,
            stage_deltas=stage_deltas,
            evidence=evidence,
        )


def score_run(
    run: RAGRun,
    oracles: Sequence[ScoreOracle],
) -> dict[str, float]:
    """Score one run with all configured oracles."""

    scores: dict[str, float] = {}
    for oracle in oracles:
        scores[oracle.name] = clamp_score(float(oracle.score(run)))
    return scores


def compute_stage_deltas(
    deltas: Mapping[str, float],
    operators: Mapping[str, MutationOperator],
) -> dict[Stage, float]:
    """Return max per-stage deltas, using zero when a stage has no signal."""

    stage_deltas: dict[Stage, float] = {}
    for stage in STAGES:
        values = [
            float(deltas[operator_id])
            for operator_id, operator in operators.items()
            if operator.stage == stage and operator_id in deltas
        ]
        stage_deltas[stage] = max(values) if values else 0.0
    return stage_deltas


def choose_stage(
    stage_deltas: Mapping[Stage, float],
    *,
    delta_threshold: float,
) -> DiagnosisStage:
    """Apply the final-plan thresholded argmax decision rule."""

    if not stage_deltas:
        return "no_fault_detected"
    best_stage = max(STAGES, key=lambda stage: stage_deltas.get(stage, 0.0))
    best_delta = stage_deltas.get(best_stage, 0.0)
    if best_delta <= delta_threshold:
        return "no_fault_detected"
    return best_stage


def confidence_for_stage(
    stage: DiagnosisStage,
    stage_deltas: Mapping[Stage, float],
) -> float:
    """Return normalized stage confidence from non-negative delta evidence."""

    if stage == "no_fault_detected":
        return 0.0
    positive_stage_deltas = {
        candidate: max(0.0, float(delta)) for candidate, delta in stage_deltas.items()
    }
    denominator = sum(positive_stage_deltas.values())
    if denominator <= 0.0:
        return 0.0
    return clamp_score(positive_stage_deltas[stage] / denominator)


def fault_report_to_dict(report: FaultReport) -> dict[str, Any]:
    """Return a JSON-friendly fault report dictionary."""

    return asdict(report)
