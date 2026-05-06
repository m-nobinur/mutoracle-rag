from __future__ import annotations

import pytest

from mutoracle.localizer.calibration import (
    CentroidDeltaCalibrator,
    LogisticDeltaCalibrator,
    NoFaultGate,
    StageThresholdCalibrator,
    tune_no_fault_gate,
    tune_stage_thresholds,
)

OPERATORS = ("A", "B", "C")


def _row(label: str, vector: tuple[float, float, float]) -> dict[str, object]:
    deltas = dict(zip(OPERATORS, vector, strict=True))
    return {
        "expected_stage": label,
        "operator_deltas": deltas,
        "stage_deltas": {
            "retrieval": vector[0],
            "prompt": vector[1],
            "generation": vector[2],
        },
    }


def _training_rows() -> list[dict[str, object]]:
    return [
        _row("retrieval", (0.8, 0.05, 0.0)),
        _row("retrieval", (0.7, 0.02, 0.0)),
        _row("prompt", (0.0, 0.7, 0.02)),
        _row("prompt", (0.0, 0.8, 0.02)),
        _row("generation", (0.0, 0.05, 0.75)),
        _row("generation", (0.0, 0.02, 0.85)),
        _row("no_fault_detected", (0.0, 0.0, 0.0)),
        _row("no_fault_detected", (0.02, 0.0, 0.01)),
    ]


def test_no_fault_gate_uses_all_explicit_conditions() -> None:
    gate = NoFaultGate(
        max_delta_threshold=0.1,
        total_positive_threshold=0.3,
        min_positive_operators=2,
    )

    assert gate.rejects_fault({"A": 0.2}, {"retrieval": 0.2}) is True
    assert gate.rejects_fault(
        {"A": 0.2, "B": 0.2},
        {"retrieval": 0.2, "prompt": 0.2},
    ) is False
    assert gate.metadata()["min_positive_operators"] == 2


def test_stage_threshold_calibrator_handles_gate_no_eligible_and_winner() -> None:
    gate = NoFaultGate(max_delta_threshold=0.05)
    calibrator = StageThresholdCalibrator(
        stage_thresholds={"retrieval": 0.5, "prompt": 0.5, "generation": 0.5},
        no_fault_gate=gate,
    )

    gated = calibrator.predict({"A": 0.01}, {"retrieval": 0.01})
    no_eligible = calibrator.predict({"A": 0.2}, {"retrieval": 0.2})
    winner = calibrator.predict({"B": 0.8}, {"prompt": 0.8})

    assert gated.stage == "no_fault_detected"
    assert no_eligible.stage == "no_fault_detected"
    assert winner.stage == "prompt"
    assert winner.confidence == pytest.approx(1.0)


def test_centroid_calibrator_predicts_full_delta_vectors_and_gate() -> None:
    gate = NoFaultGate(max_delta_threshold=0.05)
    centroid = CentroidDeltaCalibrator.fit(
        _training_rows(),
        operators=OPERATORS,
        no_fault_gate=gate,
    )

    prompt = centroid.predict(
        {"A": 0.0, "B": 0.76, "C": 0.01},
        {"prompt": 0.76},
    )
    gated = centroid.predict({"A": 0.01}, {"retrieval": 0.01})

    assert centroid.method == "zscored_nearest_centroid"
    assert prompt.stage == "prompt"
    assert prompt.confidence > 0.0
    assert gated.stage == "no_fault_detected"


def test_centroid_fit_handles_missing_labels_with_zero_centroids() -> None:
    centroid = CentroidDeltaCalibrator.fit(
        [_row("retrieval", (0.8, 0.0, 0.0))],
        operators=OPERATORS,
        no_fault_gate=NoFaultGate(max_delta_threshold=0.0),
    )

    assert centroid.centroids["prompt"] == (0.0, 0.0, 0.0)


def test_logistic_calibrator_trains_predicts_and_validates_empty_rows() -> None:
    gate = NoFaultGate(max_delta_threshold=0.05)
    logistic = LogisticDeltaCalibrator.fit(
        _training_rows(),
        operators=OPERATORS,
        no_fault_gate=gate,
        epochs=80,
        learning_rate=0.2,
    )

    prediction = logistic.predict(
        {"A": 0.0, "B": 0.02, "C": 0.8},
        {"generation": 0.8},
    )
    gated = logistic.predict({"A": 0.01}, {"retrieval": 0.01})

    assert logistic.method == "multinomial_logistic_delta"
    assert prediction.stage == "generation"
    assert prediction.scores["generation"] > 0.25
    assert gated.stage == "no_fault_detected"
    with pytest.raises(ValueError, match="no validation rows"):
        LogisticDeltaCalibrator.fit(
            [],
            operators=OPERATORS,
            no_fault_gate=gate,
        )


def test_validation_tuners_return_gate_and_stage_thresholds() -> None:
    rows = _training_rows()
    gate = tune_no_fault_gate(rows, candidates=[0.0, 0.05, 0.1])
    stage_thresholds = tune_stage_thresholds(
        rows,
        no_fault_gate=gate,
        candidates=[0.0, 0.1, 0.5],
    )

    assert gate.max_delta_threshold >= 0.0
    assert set(stage_thresholds.stage_thresholds) == {
        "retrieval",
        "prompt",
        "generation",
    }
    with pytest.raises(ValueError, match="no candidates"):
        tune_stage_thresholds(rows, no_fault_gate=gate, candidates=[])
