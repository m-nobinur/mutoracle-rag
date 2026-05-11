"""Deterministic statistics helpers for Phase 9 analysis assets."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from random import Random
from statistics import fmean
from typing import Any


@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval for a scalar metric."""

    n: int
    estimate: float
    lower: float
    upper: float


def mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty input."""

    selected = [float(value) for value in values]
    if not selected:
        return 0.0
    return float(fmean(selected))


def sample_stddev(values: Iterable[float]) -> float:
    """Return sample standard deviation, or 0.0 when fewer than two values exist."""

    selected = [float(value) for value in values]
    if len(selected) < 2:
        return 0.0
    average = fmean(selected)
    variance = sum((value - average) ** 2 for value in selected) / (len(selected) - 1)
    return sqrt(variance)


def bootstrap_ci(
    values: Sequence[Any],
    metric: Callable[[Sequence[Any]], float],
    *,
    seed: int = 2026,
    samples: int = 1000,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Return a deterministic bootstrap CI for a metric over ``values``."""

    if samples < 1:
        msg = "samples must be at least 1"
        raise ValueError(msg)
    if confidence <= 0.0 or confidence >= 1.0:
        msg = "confidence must be between 0 and 1"
        raise ValueError(msg)
    if not values:
        return BootstrapCI(n=0, estimate=0.0, lower=0.0, upper=0.0)

    rng = Random(seed)
    size = len(values)
    estimates = []
    for _ in range(samples):
        resample = [values[rng.randrange(size)] for _ in range(size)]
        estimates.append(float(metric(resample)))
    estimates.sort()
    alpha = 1.0 - confidence
    lower_index = max(0, int((alpha / 2.0) * samples))
    upper_index = min(samples - 1, int((1.0 - alpha / 2.0) * samples) - 1)
    return BootstrapCI(
        n=size,
        estimate=float(metric(values)),
        lower=estimates[lower_index],
        upper=estimates[upper_index],
    )


def accuracy_ci(
    rows: Sequence[Mapping[str, Any]],
    *,
    correct_key: str = "correct",
    seed: int = 2026,
    samples: int = 1000,
) -> BootstrapCI:
    """Return a bootstrap CI for boolean accuracy."""

    flags = [bool(row.get(correct_key, False)) for row in rows]
    return bootstrap_ci(
        flags,
        lambda selected: mean(1.0 if flag else 0.0 for flag in selected),
        seed=seed,
        samples=samples,
    )


def binary_classification_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_key: str,
    predicted_key: str,
    positive_label: str,
) -> dict[str, float]:
    """Return binary classification metrics for a positive class."""

    if not rows:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "balanced_accuracy": 0.0,
            "mcc": 0.0,
        }
    tp = fp = tn = fn = 0
    for row in rows:
        expected = str(row.get(expected_key, ""))
        predicted = str(row.get(predicted_key, ""))
        if expected == positive_label and predicted == positive_label:
            tp += 1
        elif expected != positive_label and predicted == positive_label:
            fp += 1
        elif expected == positive_label and predicted != positive_label:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mcc_denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denominator if mcc_denominator else 0.0
    return {
        "accuracy": (tp + tn) / len(rows),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "mcc": mcc,
    }


def confusion_matrix(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_key: str,
    predicted_key: str,
    labels: Sequence[str],
) -> list[list[int]]:
    """Return a square confusion matrix in the order supplied by ``labels``."""

    index = {label: offset for offset, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for row in rows:
        expected = str(row.get(expected_key, ""))
        predicted = str(row.get(predicted_key, ""))
        if expected in index and predicted in index:
            matrix[index[expected]][index[predicted]] += 1
    return matrix
