"""Validation-trained delta-vector fault localizers."""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from mutoracle.contracts import DiagnosisStage, Stage
from mutoracle.mutations import list_operator_ids
from mutoracle.oracles import clamp_score

LABELS: tuple[DiagnosisStage, ...] = (
    "retrieval",
    "prompt",
    "generation",
    "no_fault_detected",
)
STAGES: tuple[Stage, ...] = ("retrieval", "prompt", "generation")


@dataclass(frozen=True)
class DeltaPrediction:
    """One calibrated prediction over a mutation-delta vector."""

    stage: DiagnosisStage
    confidence: float
    scores: dict[str, float]
    metadata: dict[str, Any]


class DeltaVectorCalibrator(Protocol):
    """Protocol implemented by validation-trained delta-vector localizers."""

    @property
    def method(self) -> str:
        """Return the stable calibration method identifier."""

    def predict(
        self,
        deltas: Mapping[str, float],
        stage_deltas: Mapping[Stage, float],
    ) -> DeltaPrediction:
        """Predict a stage from full operator and stage deltas."""


@dataclass(frozen=True)
class NoFaultGate:
    """Explicit validation-tuned gate for no-fault examples."""

    max_delta_threshold: float
    total_positive_threshold: float = 0.0
    min_positive_operators: int = 1

    def rejects_fault(
        self,
        deltas: Mapping[str, float],
        stage_deltas: Mapping[Stage, float],
    ) -> bool:
        """Return true when mutation evidence is too weak to localize a fault."""

        max_delta = max((float(value) for value in stage_deltas.values()), default=0.0)
        total_positive = sum(max(0.0, float(value)) for value in deltas.values())
        positive_operators = sum(1 for value in deltas.values() if float(value) > 0.0)
        return (
            max_delta <= self.max_delta_threshold
            or total_positive <= self.total_positive_threshold
            or positive_operators < self.min_positive_operators
        )

    def metadata(self) -> dict[str, float | int]:
        """Return JSON-friendly gate metadata."""

        return {
            "max_delta_threshold": self.max_delta_threshold,
            "total_positive_threshold": self.total_positive_threshold,
            "min_positive_operators": self.min_positive_operators,
        }


@dataclass(frozen=True)
class StageThresholdCalibrator:
    """Validation-tuned per-stage thresholds for the transparent delta rule."""

    stage_thresholds: dict[Stage, float]
    no_fault_gate: NoFaultGate
    method: str = "stage_threshold_delta"

    def predict(
        self,
        deltas: Mapping[str, float],
        stage_deltas: Mapping[Stage, float],
    ) -> DeltaPrediction:
        if self.no_fault_gate.rejects_fault(deltas, stage_deltas):
            return DeltaPrediction(
                stage="no_fault_detected",
                confidence=0.0,
                scores={},
                metadata={"gate": self.no_fault_gate.metadata()},
            )

        eligible = {
            stage: float(stage_deltas.get(stage, 0.0))
            for stage in STAGES
            if float(stage_deltas.get(stage, 0.0))
            > float(self.stage_thresholds.get(stage, 0.0))
        }
        if not eligible:
            return DeltaPrediction(
                stage="no_fault_detected",
                confidence=0.0,
                scores={},
                metadata={
                    "gate": self.no_fault_gate.metadata(),
                    "stage_thresholds": dict(self.stage_thresholds),
                },
            )
        stage = max(eligible, key=eligible.__getitem__)
        positives = {name: max(0.0, value) for name, value in eligible.items()}
        denominator = sum(positives.values())
        confidence = positives[stage] / denominator if denominator > 0 else 0.0
        return DeltaPrediction(
            stage=stage,
            confidence=clamp_score(confidence),
            scores={name: float(value) for name, value in eligible.items()},
            metadata={
                "gate": self.no_fault_gate.metadata(),
                "stage_thresholds": dict(self.stage_thresholds),
            },
        )


@dataclass(frozen=True)
class CentroidDeltaCalibrator:
    """Nearest-centroid classifier over the standardized operator-delta vector."""

    operators: tuple[str, ...]
    means: tuple[float, ...]
    stdevs: tuple[float, ...]
    centroids: dict[DiagnosisStage, tuple[float, ...]]
    no_fault_gate: NoFaultGate
    method: str = "zscored_nearest_centroid"

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        operators: Sequence[str] | None = None,
        no_fault_gate: NoFaultGate,
    ) -> CentroidDeltaCalibrator:
        resolved_operators = tuple(operators or list_operator_ids())
        vectors = [_operator_vector(row, resolved_operators) for row in rows]
        means, stdevs = _standardizer(vectors)
        centroids: dict[DiagnosisStage, tuple[float, ...]] = {}
        for label in LABELS:
            selected = [
                _zscore(_operator_vector(row, resolved_operators), means, stdevs)
                for row in rows
                if row["expected_stage"] == label
            ]
            if not selected:
                centroids[label] = tuple(0.0 for _ in resolved_operators)
                continue
            centroids[label] = tuple(
                statistics.mean(vector[index] for vector in selected)
                for index in range(len(resolved_operators))
            )
        return cls(
            operators=resolved_operators,
            means=tuple(means),
            stdevs=tuple(stdevs),
            centroids=centroids,
            no_fault_gate=no_fault_gate,
        )

    def predict(
        self,
        deltas: Mapping[str, float],
        stage_deltas: Mapping[Stage, float],
    ) -> DeltaPrediction:
        if self.no_fault_gate.rejects_fault(deltas, stage_deltas):
            return DeltaPrediction(
                stage="no_fault_detected",
                confidence=0.0,
                scores={},
                metadata={"gate": self.no_fault_gate.metadata()},
            )

        vector = _zscore(
            [float(deltas.get(operator, 0.0)) for operator in self.operators],
            self.means,
            self.stdevs,
        )
        distances = {
            label: _squared_distance(vector, centroid)
            for label, centroid in self.centroids.items()
        }
        stage = min(distances, key=distances.__getitem__)
        ordered = sorted(distances.values())
        margin = ordered[1] - ordered[0] if len(ordered) > 1 else 0.0
        return DeltaPrediction(
            stage=stage,
            confidence=clamp_score(1.0 - math.exp(-max(0.0, margin))),
            scores={label: -distance for label, distance in distances.items()},
            metadata={"gate": self.no_fault_gate.metadata()},
        )


@dataclass(frozen=True)
class LogisticDeltaCalibrator:
    """Small multinomial logistic classifier over full operator deltas."""

    operators: tuple[str, ...]
    means: tuple[float, ...]
    stdevs: tuple[float, ...]
    weights: dict[DiagnosisStage, tuple[float, ...]]
    no_fault_gate: NoFaultGate
    method: str = "multinomial_logistic_delta"

    @classmethod
    def fit(
        cls,
        rows: Sequence[Mapping[str, Any]],
        *,
        operators: Sequence[str] | None = None,
        no_fault_gate: NoFaultGate,
        epochs: int = 500,
        learning_rate: float = 0.08,
        l2: float = 0.01,
    ) -> LogisticDeltaCalibrator:
        if not rows:
            msg = "Cannot fit logistic calibrator with no validation rows."
            raise ValueError(msg)
        resolved_operators = tuple(operators or list_operator_ids())
        raw_vectors = [_operator_vector(row, resolved_operators) for row in rows]
        means, stdevs = _standardizer(raw_vectors)
        vectors = [_with_bias(_zscore(vector, means, stdevs)) for vector in raw_vectors]
        labels = [str(row["expected_stage"]) for row in rows]
        weights = {
            label: [0.0 for _ in range(len(resolved_operators) + 1)] for label in LABELS
        }

        scale = 1.0 / len(vectors)
        for _ in range(max(1, epochs)):
            gradients = {
                label: [0.0 for _ in range(len(resolved_operators) + 1)]
                for label in LABELS
            }
            for vector, gold_label in zip(vectors, labels, strict=True):
                probabilities = _softmax(
                    {label: _dot(weights[label], vector) for label in LABELS}
                )
                for label in LABELS:
                    target = 1.0 if label == gold_label else 0.0
                    error = probabilities[label] - target
                    for index, value in enumerate(vector):
                        gradients[label][index] += error * value

            for label in LABELS:
                for index in range(len(weights[label])):
                    penalty = l2 * weights[label][index] if index > 0 else 0.0
                    weights[label][index] -= learning_rate * (
                        gradients[label][index] * scale + penalty
                    )

        return cls(
            operators=resolved_operators,
            means=tuple(means),
            stdevs=tuple(stdevs),
            weights={label: tuple(values) for label, values in weights.items()},
            no_fault_gate=no_fault_gate,
        )

    def predict(
        self,
        deltas: Mapping[str, float],
        stage_deltas: Mapping[Stage, float],
    ) -> DeltaPrediction:
        if self.no_fault_gate.rejects_fault(deltas, stage_deltas):
            return DeltaPrediction(
                stage="no_fault_detected",
                confidence=0.0,
                scores={},
                metadata={"gate": self.no_fault_gate.metadata()},
            )

        vector = _with_bias(
            _zscore(
                [float(deltas.get(operator, 0.0)) for operator in self.operators],
                self.means,
                self.stdevs,
            )
        )
        probabilities = _softmax(
            {label: _dot(self.weights[label], vector) for label in LABELS}
        )
        stage = max(probabilities, key=probabilities.__getitem__)
        return DeltaPrediction(
            stage=stage,
            confidence=clamp_score(probabilities[stage]),
            scores={label: float(value) for label, value in probabilities.items()},
            metadata={"gate": self.no_fault_gate.metadata()},
        )


def tune_no_fault_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[float] | None = None,
) -> NoFaultGate:
    """Tune the no-fault max-delta gate on validation rows."""

    thresholds = _threshold_candidates(rows) if candidates is None else list(candidates)
    best_threshold = 0.0
    best_score = -1.0
    for threshold in thresholds:
        score = _gate_balanced_accuracy(rows, threshold)
        if (score, threshold) > (best_score, best_threshold):
            best_score = score
            best_threshold = threshold
    return NoFaultGate(max_delta_threshold=best_threshold)


def tune_stage_thresholds(
    rows: Sequence[Mapping[str, Any]],
    *,
    no_fault_gate: NoFaultGate,
    candidates: Sequence[float] | None = None,
) -> StageThresholdCalibrator:
    """Tune per-stage delta thresholds on validation rows."""

    thresholds = _threshold_candidates(rows) if candidates is None else list(candidates)
    best: tuple[float, dict[Stage, float]] | None = None
    for retrieval_threshold in thresholds:
        for prompt_threshold in thresholds:
            for generation_threshold in thresholds:
                stage_thresholds: dict[Stage, float] = {
                    "retrieval": retrieval_threshold,
                    "prompt": prompt_threshold,
                    "generation": generation_threshold,
                }
                calibrator = StageThresholdCalibrator(
                    stage_thresholds=stage_thresholds,
                    no_fault_gate=no_fault_gate,
                )
                accuracy = _accuracy(rows, calibrator)
                candidate = (accuracy, stage_thresholds)
                if best is None or _threshold_tiebreak(candidate, best):
                    best = candidate
    if best is None:
        msg = "Cannot tune stage thresholds with no candidates."
        raise ValueError(msg)
    return StageThresholdCalibrator(
        stage_thresholds=best[1],
        no_fault_gate=no_fault_gate,
    )


def _operator_vector(row: Mapping[str, Any], operators: Sequence[str]) -> list[float]:
    deltas = row["operator_deltas"]
    return [float(deltas.get(operator, 0.0)) for operator in operators]


def _standardizer(
    vectors: Sequence[Sequence[float]],
) -> tuple[list[float], list[float]]:
    if not vectors:
        msg = "Cannot standardize an empty vector set."
        raise ValueError(msg)
    means = [
        statistics.mean(vector[index] for vector in vectors)
        for index in range(len(vectors[0]))
    ]
    stdevs: list[float] = []
    for index in range(len(vectors[0])):
        stdev = statistics.pstdev(vector[index] for vector in vectors)
        stdevs.append(stdev if stdev > 0.0 else 1.0)
    return means, stdevs


def _zscore(
    vector: Sequence[float],
    means: Sequence[float],
    stdevs: Sequence[float],
) -> list[float]:
    return [
        (value - mean) / stdev
        for value, mean, stdev in zip(vector, means, stdevs, strict=True)
    ]


def _with_bias(vector: Sequence[float]) -> list[float]:
    return [1.0, *vector]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right, strict=True))


def _softmax(scores: Mapping[DiagnosisStage, float]) -> dict[DiagnosisStage, float]:
    max_score = max(scores.values())
    exps = {label: math.exp(score - max_score) for label, score in scores.items()}
    denominator = sum(exps.values())
    return {label: value / denominator for label, value in exps.items()}


def _threshold_candidates(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    values = {
        round(
            max(
                (float(value) for value in row["stage_deltas"].values()),
                default=0.0,
            ),
            6,
        )
        for row in rows
    }
    return sorted({0.0, 0.01, 0.03, 0.05, 0.08, 0.1, *values})


def _gate_balanced_accuracy(
    rows: Sequence[Mapping[str, Any]],
    threshold: float,
) -> float:
    labels = list(LABELS)
    recalls: list[float] = []
    for label in labels:
        selected = [row for row in rows if row["expected_stage"] == label]
        if not selected:
            continue
        correct = 0
        for row in selected:
            max_delta = max(
                (float(value) for value in row["stage_deltas"].values()),
                default=0.0,
            )
            predicted_no_fault = max_delta <= threshold
            if predicted_no_fault == (label == "no_fault_detected"):
                correct += 1
        recalls.append(correct / len(selected))
    return statistics.mean(recalls) if recalls else 0.0


def _accuracy(
    rows: Sequence[Mapping[str, Any]],
    calibrator: DeltaVectorCalibrator,
) -> float:
    if not rows:
        return 0.0
    correct = 0
    for row in rows:
        prediction = calibrator.predict(row["operator_deltas"], row["stage_deltas"])
        if prediction.stage == row["expected_stage"]:
            correct += 1
    return correct / len(rows)


def _threshold_tiebreak(
    candidate: tuple[float, Mapping[Stage, float]],
    current: tuple[float, Mapping[Stage, float]],
) -> bool:
    candidate_sum = sum(candidate[1].values())
    current_sum = sum(current[1].values())
    return (candidate[0], candidate_sum) > (current[0], current_sum)
