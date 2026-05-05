"""Validation-only threshold tuning for baseline detectors."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise

from pydantic import BaseModel, ConfigDict, Field

from mutoracle.baselines.schema import BaselineLabel


class LabeledScore(BaseModel):
    """One validation score used for threshold selection."""

    model_config = ConfigDict(extra="forbid")

    score: float = Field(ge=0.0, le=1.0)
    expected_label: BaselineLabel
    split: str = "validation"


class ThresholdCalibration(BaseModel):
    """Selected threshold and validation metrics."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(ge=0.0, le=1.0)
    validation_f1: float = Field(ge=0.0, le=1.0)
    positives: int
    examples: int


def tune_threshold_validation_only(
    scores: Sequence[LabeledScore],
) -> ThresholdCalibration:
    """Tune a hallucination threshold using validation examples only."""

    if not scores:
        msg = "Threshold calibration requires validation scores."
        raise ValueError(msg)
    non_validation = [item.split for item in scores if item.split != "validation"]
    if non_validation:
        msg = "Threshold calibration can only consume validation split scores."
        raise ValueError(msg)

    candidates = _threshold_candidates([item.score for item in scores])
    best_threshold = candidates[0]
    best_f1 = -1.0
    for threshold in candidates:
        f1 = _hallucination_f1(scores, threshold=threshold)
        if f1 > best_f1 or (f1 == best_f1 and threshold < best_threshold):
            best_f1 = f1
            best_threshold = threshold

    positives = sum(1 for item in scores if item.expected_label == "hallucinated")
    return ThresholdCalibration(
        threshold=best_threshold,
        validation_f1=max(0.0, best_f1),
        positives=positives,
        examples=len(scores),
    )


def _threshold_candidates(scores: Sequence[float]) -> list[float]:
    unique = sorted({0.0, 1.0, *scores})
    candidates = set(unique)
    for left, right in pairwise(unique):
        candidates.add((left + right) / 2.0)
    return sorted(candidates)


def _hallucination_f1(scores: Sequence[LabeledScore], *, threshold: float) -> float:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for item in scores:
        predicted = "hallucinated" if item.score < threshold else "faithful"
        if predicted == "hallucinated" and item.expected_label == "hallucinated":
            true_positive += 1
        elif predicted == "hallucinated":
            false_positive += 1
        elif item.expected_label == "hallucinated":
            false_negative += 1

    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator == 0:
        return 1.0
    return (2 * true_positive) / denominator
