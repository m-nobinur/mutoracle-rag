"""Transparent mutation-delta fault localization."""

from __future__ import annotations

import hashlib
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
from mutoracle.localizer.calibration import DeltaVectorCalibrator
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
        stage_thresholds: Mapping[Stage, float] | None = None,
        calibrator: DeltaVectorCalibrator | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._oracles = list(oracles)
        if not self._oracles:
            msg = "FaultLocalizer requires at least one oracle."
            raise ValueError(msg)
        self._aggregator = aggregator
        delta_threshold = float(delta_threshold)
        if delta_threshold < 0.0 or delta_threshold > 1.0:
            msg = "delta_threshold must be in [0, 1]."
            raise ValueError(msg)
        self._delta_threshold = delta_threshold
        self._operators = dict(mutation_registry() if operators is None else operators)
        self._seed = seed
        self._stage_thresholds = (
            None if stage_thresholds is None else dict(stage_thresholds)
        )
        self._calibrator = calibrator

    def diagnose(self, query: str) -> FaultReport:
        """Return a stage-level fault report for one query."""

        baseline = self._pipeline.run(query)
        scorable_runs = [baseline]
        operator_runs: list[tuple[str, MutationOperator, RAGRun | None, bool]] = []
        operator_status: dict[str, dict[str, Any]] = {}
        evidence = [
            f"Delta threshold: {self._delta_threshold:.4f}.",
        ]

        for operator_id, operator in self._operators.items():
            rng = _operator_rng(self._seed, operator_id)
            mutated = operator.apply(baseline, rng=rng)
            mutation = mutated.metadata.get("mutation", {})
            if isinstance(mutation, dict) and mutation.get("rejected") is True:
                reason = mutation.get("rejection_reason", "mutation rejected")
                evidence.append(f"{operator_id} skipped: {reason}.")
                operator_status[operator_id] = {
                    "stage": operator.stage,
                    "applied": False,
                    "rejected": True,
                    "reason": str(reason),
                }
                operator_runs.append((operator_id, operator, None, False))
                continue

            scored_run = _materialize_scored_run(
                pipeline=self._pipeline,
                baseline=baseline,
                mutated=mutated,
                operator=operator,
            )
            if scored_run is not mutated:
                evidence.append(
                    f"{operator_id} reran pipeline with mutated query for scoring."
                )
            scorable_runs.append(scored_run)
            operator_status[operator_id] = {
                "stage": operator.stage,
                "applied": True,
                "rejected": False,
            }
            operator_runs.append((operator_id, operator, scored_run, True))

        all_scores = score_runs(scorable_runs, self._oracles)
        baseline_scores = all_scores[0]
        baseline_omega = self._aggregator.combine(baseline_scores)
        evidence.insert(0, f"Baseline composite score: {baseline_omega:.4f}.")

        deltas: dict[str, float] = {}
        score_index = 1
        for operator_id, operator, operator_run, should_score in operator_runs:
            if not should_score:
                deltas[operator_id] = 0.0
                continue
            if operator_run is None:
                msg = f"{operator_id} was marked scorable without a run."
                raise RuntimeError(msg)
            mutated_scores = all_scores[score_index]
            score_index += 1
            mutated_omega = self._aggregator.combine(mutated_scores)
            delta = baseline_omega - mutated_omega
            deltas[operator_id] = delta
            evidence.append(
                f"{operator_id} ({operator.stage}) delta: {delta:.4f} "
                f"from mutated score {mutated_omega:.4f}."
            )

        stage_deltas = compute_stage_deltas(deltas, self._operators)
        if self._calibrator is None:
            stage = choose_stage(
                stage_deltas,
                delta_threshold=self._delta_threshold,
                stage_thresholds=self._stage_thresholds,
            )
            confidence = confidence_for_stage(stage, stage_deltas)
        else:
            calibrated = self._calibrator.predict(deltas, stage_deltas)
            stage = calibrated.stage
            confidence = calibrated.confidence
            evidence.append(f"Calibrator: {self._calibrator.method}.")
            if calibrated.metadata:
                evidence.append(f"Calibration metadata: {calibrated.metadata}.")
        evidence.append(f"Predicted stage: {stage} with confidence {confidence:.4f}.")
        return FaultReport(
            stage=stage,
            confidence=confidence,
            deltas=deltas,
            stage_deltas=stage_deltas,
            operator_status=operator_status,
            evidence=evidence,
        )


def score_run(
    run: RAGRun,
    oracles: Sequence[ScoreOracle],
) -> dict[str, float]:
    """Score one run with all configured oracles."""

    return score_runs([run], oracles)[0]


def score_runs(
    runs: Sequence[RAGRun],
    oracles: Sequence[ScoreOracle],
) -> list[dict[str, float]]:
    """Score multiple runs with all configured oracles."""

    scores_by_run: list[dict[str, float]] = [dict() for _ in runs]
    if not runs:
        return scores_by_run
    for oracle in oracles:
        score_results = getattr(oracle, "score_results", None)
        if callable(score_results):
            values = [float(result.value) for result in score_results(runs)]
        else:
            values = [float(oracle.score(run)) for run in runs]
        if len(values) != len(runs):
            msg = (
                f"Oracle {oracle.name} returned {len(values)} scores "
                f"for {len(runs)} runs."
            )
            raise ValueError(msg)
        for index, value in enumerate(values):
            scores_by_run[index][oracle.name] = clamp_score(value)
    return scores_by_run


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
    stage_thresholds: Mapping[Stage, float] | None = None,
) -> DiagnosisStage:
    """Apply the final-plan thresholded argmax decision rule."""

    if not stage_deltas:
        return "no_fault_detected"
    best_stage = max(STAGES, key=lambda stage: stage_deltas.get(stage, 0.0))
    best_delta = stage_deltas.get(best_stage, 0.0)
    threshold = (
        float(delta_threshold)
        if stage_thresholds is None
        else float(stage_thresholds.get(best_stage, delta_threshold))
    )
    if best_delta <= threshold:
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


def _materialize_scored_run(
    *,
    pipeline: RAGPipeline,
    baseline: RAGRun,
    mutated: RAGRun,
    operator: MutationOperator,
) -> RAGRun:
    """Return the run representation that should be scored for one mutation."""

    if operator.stage != "prompt" or mutated.query == baseline.query:
        return mutated

    rerun = pipeline.run(mutated.query)
    mutation_record = mutated.metadata.get("mutation")
    mutation_history = mutated.metadata.get("mutations")
    metadata = dict(rerun.metadata)
    if mutation_record is not None:
        metadata["mutation"] = mutation_record
    if isinstance(mutation_history, list):
        metadata["mutations"] = mutation_history
    return RAGRun(
        query=rerun.query,
        passages=rerun.passages,
        answer=rerun.answer,
        metadata=metadata,
    )


def _operator_rng(seed: int, operator_id: str) -> Random:
    """Return a deterministic operator-scoped RNG independent of iteration order."""

    digest = hashlib.sha256(f"{seed}:{operator_id}".encode()).digest()
    return Random(int.from_bytes(digest[:8], "big", signed=False))
