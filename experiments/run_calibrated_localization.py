"""Train and compare validation-calibrated FITS localizers.

The transparent MutOracle localizer is intentionally simple: max stage delta
above a global threshold. This script keeps that rule as a baseline, then adds
explicit no-fault gating, validation-tuned stage thresholds, nearest-centroid
classification, and a small multinomial logistic classifier over the full
operator-delta vector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mutoracle.experiments import (
    FITSRecordPipeline,
    build_experiment_aggregator,
    expected_diagnosis_stage,
    fixture_oracles,
)
from mutoracle.localizer import (
    CentroidDeltaCalibrator,
    FaultLocalizer,
    LogisticDeltaCalibrator,
    StageThresholdCalibrator,
    choose_stage,
    confidence_for_stage,
    tune_no_fault_gate,
    tune_stage_thresholds,
)
from mutoracle.mutations import list_operator_ids

LABELS = ("retrieval", "prompt", "generation", "no_fault_detected")


def main() -> None:
    args = _parse_args()
    validation_rows = _validation_delta_rows(args.fits_path, seeds=args.seeds)
    no_fault_gate = tune_no_fault_gate(validation_rows)
    stage_threshold = tune_stage_thresholds(
        validation_rows,
        no_fault_gate=no_fault_gate,
    )
    centroid = CentroidDeltaCalibrator.fit(
        validation_rows,
        no_fault_gate=no_fault_gate,
    )
    logistic = LogisticDeltaCalibrator.fit(
        validation_rows,
        no_fault_gate=no_fault_gate,
        epochs=args.logistic_epochs,
    )
    models: list[tuple[str, Any]] = [
        ("transparent_global_threshold", None),
        ("no_fault_gated_delta", _NoFaultGatedTransparent(no_fault_gate)),
        ("stage_threshold_delta", stage_threshold),
        ("centroid_full_delta", centroid),
        ("logistic_full_delta", logistic),
    ]

    test_rows = _read_jsonl(args.e2_raw)
    calibrated_rows = [
        _calibrated_row(row, name, model) for name, model in models for row in test_rows
    ]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = output_dir / "e2_localization_calibrated_full_raw.jsonl"
    manifest_json = output_dir / "e2_localization_calibrated_full_manifest.json"
    summary_csv = output_dir / "e2_localization_calibrated_full_summary.csv"
    model_json = output_dir / "e2_localization_calibrated_full_models.json"

    _write_jsonl(raw_jsonl, calibrated_rows)
    _write_summary(summary_csv, calibrated_rows)
    _write_model_metadata(
        model_json,
        validation_rows=validation_rows,
        models=models,
        no_fault_gate=no_fault_gate,
    )
    _write_manifest(
        manifest_json,
        raw_jsonl=raw_jsonl,
        summary_csv=summary_csv,
        model_json=model_json,
        validation_rows=validation_rows,
        calibrated_rows=calibrated_rows,
        e2_raw=args.e2_raw,
        fits_path=args.fits_path,
        seeds=args.seeds,
    )
    print(f"Wrote {len(calibrated_rows)} calibrated rows to {raw_jsonl}")
    print(f"Wrote manifest to {manifest_json}")
    for line in _summary_lines(calibrated_rows):
        print(line)


class _NoFaultGatedTransparent:
    method = "no_fault_gated_delta"

    def __init__(self, no_fault_gate: Any) -> None:
        self._no_fault_gate = no_fault_gate

    def predict(self, deltas: Mapping[str, float], stage_deltas: Mapping[str, float]):
        if self._no_fault_gate.rejects_fault(deltas, stage_deltas):
            return type(
                "Prediction",
                (),
                {
                    "stage": "no_fault_detected",
                    "confidence": 0.0,
                    "metadata": {"gate": self._no_fault_gate.metadata()},
                },
            )()
        stage = choose_stage(stage_deltas, delta_threshold=0.05)
        return type(
            "Prediction",
            (),
            {
                "stage": stage,
                "confidence": confidence_for_stage(stage, stage_deltas),
                "metadata": {"gate": self._no_fault_gate.metadata()},
            },
        )()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fits-path",
        type=Path,
        default=Path("data/fits/fits_v1.0.0/fits.jsonl"),
    )
    parser.add_argument(
        "--e2-raw",
        type=Path,
        default=Path(
            "experiments/results/e2_localization/e2_localization_full_raw.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/e2_localization_calibrated"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 91])
    parser.add_argument("--logistic-epochs", type=int, default=500)
    return parser.parse_args()


def _validation_delta_rows(path: Path, *, seeds: Sequence[int]) -> list[dict[str, Any]]:
    records = [
        row for row in _read_jsonl(path) if str(row.get("split", "")) == "validation"
    ]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for record in records:
            localizer = FaultLocalizer(
                pipeline=FITSRecordPipeline(record, seed=seed),
                oracles=fixture_oracles(["nli", "semantic_similarity", "llm_judge"]),
                aggregator=build_experiment_aggregator(
                    strategy="weighted",
                    weights={
                        "nli": 0.4,
                        "semantic_similarity": 0.3,
                        "llm_judge": 0.3,
                    },
                ),
                delta_threshold=0.05,
                seed=seed,
            )
            report = localizer.diagnose(str(record["query"]))
            rows.append(
                {
                    "qid": record["qid"],
                    "seed": seed,
                    "expected_stage": expected_diagnosis_stage(record),
                    "operator_deltas": report.deltas,
                    "stage_deltas": report.stage_deltas,
                }
            )
    return rows


def _calibrated_row(
    row: Mapping[str, Any],
    localizer_name: str,
    model: Any,
) -> dict[str, Any]:
    if model is None:
        predicted = str(row["predicted_stage"])
        confidence = float(row.get("confidence", 0.0))
        metadata = {"method": "transparent_global_threshold", "delta_threshold": 0.05}
    else:
        prediction = model.predict(row["operator_deltas"], row["stage_deltas"])
        predicted = str(prediction.stage)
        confidence = float(prediction.confidence)
        metadata = dict(prediction.metadata)
        metadata["method"] = getattr(model, "method", localizer_name)
    expected = str(row["expected_stage"])
    return {
        "experiment_id": "e2_localization_calibrated",
        "mode": "full",
        "localizer_name": localizer_name,
        "qid": row["qid"],
        "seed": row["seed"],
        "split": row.get("split", "test"),
        "fault_stage": row.get("fault_stage"),
        "expected_stage": expected,
        "predicted_stage": predicted,
        "correct": predicted == expected,
        "confidence": round(confidence, 6),
        "operator_deltas": row["operator_deltas"],
        "stage_deltas": row["stage_deltas"],
        "operator_status": row.get("operator_status", {}),
        "calibration": metadata,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_summary(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = ["localizer,group,examples,correct,accuracy"]
    for localizer_name in sorted({str(row["localizer_name"]) for row in rows}):
        localizer_rows = [
            row for row in rows if str(row["localizer_name"]) == localizer_name
        ]
        for group in ["all", *LABELS]:
            selected = (
                localizer_rows
                if group == "all"
                else [
                    row for row in localizer_rows if row["expected_stage"] == group
                ]
            )
            total = len(selected)
            correct = sum(1 for row in selected if row["correct"] is True)
            accuracy = correct / total if total else 0.0
            lines.append(f"{localizer_name},{group},{total},{correct},{accuracy:.6f}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_model_metadata(
    path: Path,
    *,
    validation_rows: Sequence[Mapping[str, Any]],
    models: Sequence[tuple[str, Any]],
    no_fault_gate: Any,
) -> None:
    payload = {
        "operators": list_operator_ids(),
        "validation_rows": len(validation_rows),
        "no_fault_gate": no_fault_gate.metadata(),
        "models": [_model_payload(name, model) for name, model in models],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")


def _model_payload(name: str, model: Any) -> dict[str, Any]:
    payload = {
        "name": name,
        "method": "transparent_global_threshold"
        if model is None
        else getattr(model, "method", name),
    }
    if isinstance(model, StageThresholdCalibrator):
        payload["stage_thresholds"] = model.stage_thresholds
    return payload


def _write_manifest(
    path: Path,
    *,
    raw_jsonl: Path,
    summary_csv: Path,
    model_json: Path,
    validation_rows: Sequence[Mapping[str, Any]],
    calibrated_rows: Sequence[Mapping[str, Any]],
    e2_raw: Path,
    fits_path: Path,
    seeds: Sequence[int],
) -> None:
    payload = {
        "experiment_id": "e2_localization_calibrated",
        "mode": "full",
        "run_id": _run_id(raw_jsonl, validation_rows, calibrated_rows),
        "git_commit": _git_commit(),
        "raw_jsonl": str(raw_jsonl),
        "summary_csv": str(summary_csv),
        "model_json": str(model_json),
        "source_e2_raw": str(e2_raw),
        "fits_path": str(fits_path),
        "seeds": list(seeds),
        "row_count": len(calibrated_rows),
        "validation_row_count": len(validation_rows),
        "status": "complete",
        "metadata": {
            "script": "experiments/run_calibrated_localization.py",
            "methods": sorted(
                {str(row["localizer_name"]) for row in calibrated_rows}
            ),
            "training_split": "fits_validation",
            "test_source": "e2_localization_full_raw",
            "uses_test_labels_for_calibration": False,
        },
        "written_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", "utf-8")


def _run_id(
    raw_jsonl: Path,
    validation_rows: Sequence[Mapping[str, Any]],
    calibrated_rows: Sequence[Mapping[str, Any]],
) -> str:
    digest = hashlib.sha256()
    digest.update(str(raw_jsonl).encode())
    for collection in (validation_rows, calibrated_rows):
        digest.update(json.dumps(list(collection), sort_keys=True).encode())
    return digest.hexdigest()[:16]


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _summary_lines(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for localizer_name in sorted({str(row["localizer_name"]) for row in rows}):
        selected = [row for row in rows if row["localizer_name"] == localizer_name]
        accuracy = statistics.mean(1.0 if row["correct"] else 0.0 for row in selected)
        lines.append(f"{localizer_name}: accuracy={accuracy:.3f} n={len(selected)}")
    return lines


if __name__ == "__main__":
    main()
