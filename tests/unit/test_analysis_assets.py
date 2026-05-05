from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path.cwd()))

from experiments.analyze_results import AnalysisError, analyze_results
from experiments.stats import accuracy_ci, binary_classification_metrics


def test_bootstrap_ci_is_deterministic() -> None:
    rows = [{"correct": True}, {"correct": False}, {"correct": True}]

    first = accuracy_ci(rows, seed=17, samples=50)
    second = accuracy_ci(rows, seed=17, samples=50)

    assert first == second
    assert first.estimate == pytest.approx(2 / 3)


def test_binary_classification_metrics() -> None:
    rows = [
        {"expected": "hallucinated", "predicted": "hallucinated"},
        {"expected": "faithful", "predicted": "hallucinated"},
        {"expected": "hallucinated", "predicted": "faithful"},
        {"expected": "faithful", "predicted": "faithful"},
    ]

    metrics = binary_classification_metrics(
        rows,
        expected_key="expected",
        predicted_key="predicted",
        positive_label="hallucinated",
    )

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)


def test_analysis_imports_empty_duckdb_and_generates_assets(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_phase_nine_fixture_results(results_dir)

    outputs = analyze_results(
        results_dir=results_dir,
        paper_dir=tmp_path / "paper",
        bootstrap_samples=25,
        duckdb_path=tmp_path / "paper" / "analysis.duckdb",
    )

    table_names = {path.name for path in outputs.tables}
    figure_names = {path.name for path in outputs.figures}
    assert table_names == {
        "tab_detection.tex",
        "tab_localization.tex",
        "tab_oracle_ablation.tex",
        "tab_mutation_discriminativeness.tex",
        "tab_latency_cost.tex",
        "tab_aggregation.tex",
    }
    assert figure_names == {
        "fig_architecture.svg",
        "fig_delta_heatmap.svg",
        "fig_weight_sensitivity_heatmap.svg",
        "fig_confusion_matrix.svg",
        "fig_latency_cost.svg",
    }
    assert outputs.duckdb_path is not None
    assert outputs.duckdb_path.exists()

    detection = (tmp_path / "paper" / "tables" / "tab_detection.tex").read_text(
        encoding="utf-8"
    )
    assert "Run IDs" in detection
    assert "run-e1" in detection

    traceability = outputs.traceability.read_text(encoding="utf-8")
    assert "Cell provenance" in traceability
    assert "run-e2" in traceability


def test_missing_result_files_fail_clearly(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_result(
        results_dir,
        "e1_detection",
        [{"experiment_id": "e1_detection", "baseline_name": "ragas"}],
    )

    with pytest.raises(AnalysisError, match="Missing required result files"):
        analyze_results(results_dir=results_dir, paper_dir=tmp_path / "paper")


def test_missing_experiment_rows_fail_clearly(tmp_path: Path) -> None:
    results_dir = tmp_path / "results"
    _write_phase_nine_fixture_results(results_dir)

    e2_raw = results_dir / "e2_localization" / "e2_localization_smoke_raw.jsonl"
    rewritten = []
    for line in e2_raw.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["experiment_id"] = "mismatched_experiment"
        rewritten.append(json.dumps(row, sort_keys=True))
    e2_raw.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

    with pytest.raises(
        AnalysisError,
        match="Imported raw results are missing rows for required experiments",
    ):
        analyze_results(results_dir=results_dir, paper_dir=tmp_path / "paper")


def test_generated_figures_fit_column_width_and_have_legible_text(
    tmp_path: Path,
) -> None:
    results_dir = tmp_path / "results"
    _write_phase_nine_fixture_results(results_dir)

    outputs = analyze_results(results_dir=results_dir, paper_dir=tmp_path / "paper")

    for figure_path in outputs.figures:
        svg = figure_path.read_text(encoding="utf-8")
        width_match = re.search(r'width="([0-9]+(?:\\.[0-9]+)?)"', svg)
        assert width_match is not None
        assert float(width_match.group(1)) <= 450.0

        font_sizes = [
            float(match.group(1))
            for match in re.finditer(r'font-size="([0-9]+(?:\\.[0-9]+)?)"', svg)
        ]
        assert font_sizes
        assert min(font_sizes) >= 11.0


def _write_phase_nine_fixture_results(results_dir: Path) -> None:
    _write_result(
        results_dir,
        "e1_detection",
        [
            {
                "experiment_id": "e1_detection",
                "baseline_name": "ragas",
                "expected_label": "hallucinated",
                "predicted_label": "hallucinated",
                "correct": True,
            },
            {
                "experiment_id": "e1_detection",
                "baseline_name": "metarag",
                "expected_label": "faithful",
                "predicted_label": "faithful",
                "correct": True,
            },
        ],
    )
    _write_result(
        results_dir,
        "e2_localization",
        [
            {
                "experiment_id": "e2_localization",
                "expected_stage": "retrieval",
                "predicted_stage": "retrieval",
                "correct": True,
            },
            {
                "experiment_id": "e2_localization",
                "expected_stage": "generation",
                "predicted_stage": "prompt",
                "correct": False,
            },
        ],
    )
    _write_result(
        results_dir,
        "e3_ablation",
        [
            {
                "experiment_id": "e3_ablation",
                "ablation_name": "nli_only",
                "oracles": ["nli"],
                "correct": True,
            }
        ],
    )
    _write_result(
        results_dir,
        "e4_separability",
        [
            {
                "experiment_id": "e4_separability",
                "expected_stage": "retrieval",
                "operator_deltas": {"CI": 0.1, "CR": 0.3, "QP": 0.0},
            }
        ],
    )
    _write_result(
        results_dir,
        "e5_latency",
        [
            {
                "experiment_id": "e5_latency",
                "workflow": "rag_fixture",
                "latency_seconds": 0.01,
                "cost_usd": 0.0,
                "overhead_vs_rag": 1.0,
            },
            {
                "experiment_id": "e5_latency",
                "workflow": "mutoracle_localizer",
                "latency_seconds": 0.03,
                "cost_usd": 0.02,
                "overhead_vs_rag": 3.0,
            },
        ],
    )
    _write_result(
        results_dir,
        "e6_weighted",
        [
            {
                "experiment_id": "e6_weighted",
                "ablation_name": "weighted_calibrated",
                "aggregation": "weighted",
                "correct": True,
                "confidence": 0.8,
            },
            {
                "experiment_id": "e6_weighted",
                "ablation_name": "uniform",
                "aggregation": "uniform",
                "correct": False,
                "confidence": 0.4,
            },
        ],
    )


def _write_result(
    results_dir: Path,
    experiment_id: str,
    rows: list[dict[str, object]],
) -> None:
    output_dir = results_dir / experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl = output_dir / f"{experiment_id}_smoke_raw.jsonl"
    raw_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "experiment_id": experiment_id,
        "mode": "smoke",
        "run_id": f"run-{experiment_id.split('_')[0]}",
        "git_commit": "abc123",
        "raw_jsonl": str(raw_jsonl),
    }
    (output_dir / f"{experiment_id}_smoke_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True),
        encoding="utf-8",
    )
