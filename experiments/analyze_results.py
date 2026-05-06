"""Generate Phase 9 paper tables, figures, and provenance from saved results."""

from __future__ import annotations

import argparse
import html
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

try:
    from experiments.stats import (
        accuracy_ci,
        binary_classification_metrics,
        bootstrap_ci,
        confusion_matrix,
        mean,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution path.
    from stats import (  # type: ignore[no-redef]
        accuracy_ci,
        binary_classification_metrics,
        bootstrap_ci,
        confusion_matrix,
        mean,
    )

REQUIRED_EXPERIMENTS = (
    "e1_detection",
    "e2_localization",
    "e3_ablation",
    "e4_separability",
    "e5_latency",
    "e6_weighted",
)
OPTIONAL_EXPERIMENTS = ("e2_localization_calibrated",)
STAGE_LABELS = ("retrieval", "prompt", "generation", "no_fault_detected")
MUTATION_STAGES = {
    "CI": "retrieval",
    "CR": "retrieval",
    "CS": "retrieval",
    "QP": "prompt",
    "QN": "prompt",
    "QD": "prompt",
    "QI": "prompt",
    "FS": "generation",
    "FA": "generation",
    "FE": "generation",
    "GN": "generation",
}


@dataclass(frozen=True)
class ResultArtifact:
    """One Phase 8 raw result file and its manifest provenance."""

    experiment_id: str
    mode: str
    raw_jsonl: Path
    manifest_json: Path
    run_id: str
    git_commit: str


@dataclass(frozen=True)
class AnalysisOutputs:
    """Paths written by one Phase 9 analysis invocation."""

    tables: list[Path]
    figures: list[Path]
    traceability: Path
    duckdb_path: Path | None


class AnalysisError(RuntimeError):
    """Clear user-facing error for missing or inconsistent analysis inputs."""


def main() -> None:
    args = _parse_args()
    outputs = analyze_results(
        results_dir=args.results_dir,
        paper_dir=args.paper_dir,
        mode=args.mode,
        require_experiments=args.require_experiments,
        make_tables=args.tables or not args.figures,
        make_figures=args.figures or not args.tables,
        duckdb_path=args.duckdb_path,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    print(f"Wrote {len(outputs.tables)} tables to {args.paper_dir / 'tables'}")
    print(f"Wrote {len(outputs.figures)} figures to {args.paper_dir / 'figures'}")
    print(f"Wrote traceability report to {outputs.traceability}")
    if outputs.duckdb_path is not None:
        print(f"Wrote DuckDB database to {outputs.duckdb_path}")


def analyze_results(
    *,
    results_dir: Path = Path("experiments/results"),
    paper_dir: Path = Path("paper"),
    mode: str = "smoke",
    require_experiments: Sequence[str] = REQUIRED_EXPERIMENTS,
    make_tables: bool = True,
    make_figures: bool = True,
    duckdb_path: Path | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 2026,
) -> AnalysisOutputs:
    """Generate all requested Phase 9 assets from saved Phase 8 artifacts."""

    artifacts = discover_artifacts(
        results_dir=results_dir,
        mode=mode,
        require_experiments=require_experiments,
    )
    artifacts = [
        *artifacts,
        *discover_optional_artifacts(
            results_dir=results_dir,
            mode=mode,
            experiment_ids=OPTIONAL_EXPERIMENTS,
            existing_experiment_ids={artifact.experiment_id for artifact in artifacts},
        ),
    ]
    connection = connect_duckdb(duckdb_path)
    import_results(connection, artifacts)

    raw_rows = _fetch_all_dicts(connection, "SELECT * FROM raw_results")
    rows_by_experiment = _rows_by_experiment(raw_rows)
    _ensure_required_experiment_rows(
        rows_by_experiment,
        required_experiments=require_experiments,
    )
    manifests = {artifact.experiment_id: artifact for artifact in artifacts}

    table_dir = paper_dir / "tables"
    figure_dir = paper_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    trace_entries: list[dict[str, Any]] = []
    tables: list[Path] = []
    figures: list[Path] = []

    if make_tables:
        table_payloads = [
            detection_table(
                rows_by_experiment["e1_detection"],
                manifests,
                seed,
                bootstrap_samples,
            ),
            localization_table(
                rows_by_experiment["e2_localization"],
                rows_by_experiment.get("e2_localization_calibrated", []),
                manifests,
                seed,
                bootstrap_samples,
            ),
            ablation_table(
                rows_by_experiment["e3_ablation"],
                manifests,
                seed,
                bootstrap_samples,
            ),
            mutation_discriminativeness_table(
                rows_by_experiment["e4_separability"],
                manifests,
                seed,
                bootstrap_samples,
            ),
            latency_cost_table(
                rows_by_experiment["e5_latency"],
                manifests,
                seed,
                bootstrap_samples,
            ),
            aggregation_table(
                rows_by_experiment["e6_weighted"],
                manifests,
                seed,
                bootstrap_samples,
            ),
        ]
        for filename, headers, body, provenance in table_payloads:
            path = table_dir / filename
            write_latex_table(path, headers=headers, rows=body, provenance=provenance)
            tables.append(path)
            trace_entries.append(
                {
                    "asset": str(path),
                    "type": "table",
                    "run_ids": provenance["run_ids"],
                    "source_files": provenance["source_files"],
                    "mode": mode,
                    "cells": provenance["cells"],
                }
            )

    if make_figures:
        figure_payloads = [
            architecture_figure(),
            experiment_design_figure(),
            calibration_flow_figure(),
            localizer_accuracy_figure(
                rows_by_experiment["e2_localization"],
                rows_by_experiment.get("e2_localization_calibrated", []),
                manifests,
            ),
            delta_heatmap(rows_by_experiment["e4_separability"], manifests),
            weight_sensitivity_heatmap(rows_by_experiment["e6_weighted"], manifests),
            confusion_matrix_figure(
                rows_by_experiment.get("e2_localization_calibrated")
                or rows_by_experiment["e2_localization"],
                manifests,
            ),
            latency_cost_figure(rows_by_experiment["e5_latency"], manifests),
        ]
        for filename, svg, provenance in figure_payloads:
            path = figure_dir / filename
            path.write_text(svg, encoding="utf-8")
            figures.append(path)
            trace_entries.append(
                {
                    "asset": str(path),
                    "type": "figure",
                    "run_ids": provenance["run_ids"],
                    "source_files": provenance["source_files"],
                    "mode": mode,
                    "cells": provenance.get("cells", []),
                }
            )

    traceability = paper_dir / "TRACEABILITY.md"
    write_traceability_report(
        traceability,
        entries=trace_entries,
        artifacts=artifacts,
        mode=mode,
    )
    connection.close()
    return AnalysisOutputs(
        tables=tables,
        figures=figures,
        traceability=traceability,
        duckdb_path=duckdb_path,
    )


def discover_artifacts(
    *,
    results_dir: Path,
    mode: str,
    require_experiments: Sequence[str],
) -> list[ResultArtifact]:
    """Discover raw JSONL and manifest pairs for selected experiment IDs."""

    if not results_dir.exists():
        msg = f"Result directory does not exist: {results_dir}"
        raise AnalysisError(msg)

    discovered: dict[str, ResultArtifact] = {}
    for manifest_path in sorted(results_dir.glob(f"*/*_{mode}_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        experiment_id = str(manifest.get("experiment_id", "")).strip()
        if not experiment_id:
            continue
        raw_jsonl = Path(str(manifest.get("raw_jsonl", "")))
        if not raw_jsonl.is_absolute():
            raw_jsonl = Path.cwd() / raw_jsonl
        if not raw_jsonl.exists():
            msg = (
                f"Manifest {manifest_path} points to a missing raw result file: "
                f"{raw_jsonl}"
            )
            raise AnalysisError(msg)
        discovered[experiment_id] = ResultArtifact(
            experiment_id=experiment_id,
            mode=str(manifest.get("mode", mode)),
            raw_jsonl=raw_jsonl,
            manifest_json=manifest_path,
            run_id=str(manifest.get("run_id", "")),
            git_commit=str(manifest.get("git_commit", "")),
        )

    missing = [name for name in require_experiments if name not in discovered]
    if missing:
        available = ", ".join(sorted(discovered)) or "none"
        msg = (
            "Missing required result files for mode "
            f"{mode!r}: {', '.join(missing)}. Available experiments: {available}."
        )
        raise AnalysisError(msg)

    return [discovered[name] for name in require_experiments]


def discover_optional_artifacts(
    *,
    results_dir: Path,
    mode: str,
    experiment_ids: Sequence[str],
    existing_experiment_ids: set[str],
) -> list[ResultArtifact]:
    """Discover optional artifacts without making clean-clone smoke analysis fail."""

    optional: list[ResultArtifact] = []
    for experiment_id in experiment_ids:
        if experiment_id in existing_experiment_ids:
            continue
        manifest_path = (
            results_dir / experiment_id / f"{experiment_id}_{mode}_manifest.json"
        )
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_jsonl = Path(str(manifest.get("raw_jsonl", "")))
        if not raw_jsonl.is_absolute():
            raw_jsonl = Path.cwd() / raw_jsonl
        if not raw_jsonl.exists():
            msg = (
                f"Manifest {manifest_path} points to a missing raw result file: "
                f"{raw_jsonl}"
            )
            raise AnalysisError(msg)
        optional.append(
            ResultArtifact(
                experiment_id=experiment_id,
                mode=str(manifest.get("mode", mode)),
                raw_jsonl=raw_jsonl,
                manifest_json=manifest_path,
                run_id=str(manifest.get("run_id", "")),
                git_commit=str(manifest.get("git_commit", "")),
            )
        )
    return optional


def connect_duckdb(path: Path | None) -> duckdb.DuckDBPyConnection:
    """Return an in-memory or file-backed DuckDB connection."""

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(path))
    return duckdb.connect(":memory:")


def import_results(
    connection: duckdb.DuckDBPyConnection,
    artifacts: Sequence[ResultArtifact],
) -> None:
    """Import Phase 8 JSONL records and manifest provenance into DuckDB."""

    raw_files = ", ".join(
        _sql_literal(str(artifact.raw_jsonl)) for artifact in artifacts
    )
    connection.execute(
        "CREATE OR REPLACE TABLE raw_results AS "
        f"SELECT * FROM read_json_auto([{raw_files}], union_by_name=true)"
    )
    connection.execute(
        "CREATE OR REPLACE TABLE manifest_results ("
        "experiment_id VARCHAR, mode VARCHAR, run_id VARCHAR, git_commit VARCHAR, "
        "raw_jsonl VARCHAR, manifest_json VARCHAR)"
    )
    connection.executemany(
        "INSERT INTO manifest_results VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                artifact.experiment_id,
                artifact.mode,
                artifact.run_id,
                artifact.git_commit,
                str(artifact.raw_jsonl),
                str(artifact.manifest_json),
            )
            for artifact in artifacts
        ],
    )


def detection_table(
    rows: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, ResultArtifact],
    seed: int,
    bootstrap_samples: int,
) -> tuple[str, list[str], list[list[str]], dict[str, Any]]:
    """Build the response-level detection table."""

    grouped = _group_rows(rows, "baseline_name")
    table_rows: list[list[str]] = []
    cells: list[dict[str, str]] = []
    for baseline in sorted(grouped):
        selected = grouped[baseline]
        metric = binary_classification_metrics(
            selected,
            expected_key="expected_label",
            predicted_key="predicted_label",
            positive_label="hallucinated",
        )
        f1_ci = bootstrap_ci(
            list(selected),
            lambda sample: binary_classification_metrics(
                sample,
                expected_key="expected_label",
                predicted_key="predicted_label",
                positive_label="hallucinated",
            )["f1"],
            seed=seed,
            samples=bootstrap_samples,
        )
        row = [
            _display_name(baseline),
            str(len(selected)),
            _pct(metric["accuracy"]),
            _pct(metric["precision"]),
            _pct(metric["recall"]),
            _ci_text(f1_ci.estimate, f1_ci.lower, f1_ci.upper),
        ]
        table_rows.append(row)
        cells.append(_cell("detection", baseline, "f1", manifests["e1_detection"]))
    provenance = _provenance([manifests["e1_detection"]], cells)
    return (
        "tab_detection.tex",
        ["Method", "N", "Acc.", "Prec.", "Rec.", "F1 (95\\% CI)"],
        table_rows,
        provenance,
    )


def localization_table(
    rows: Sequence[Mapping[str, Any]],
    calibrated_rows: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, ResultArtifact],
    seed: int,
    bootstrap_samples: int,
) -> tuple[str, list[str], list[list[str]], dict[str, Any]]:
    """Build the FITS stage-attribution table."""

    table_rows: list[list[str]] = []
    cells: list[dict[str, str]] = []
    variants = [("Transparent Delta", list(rows), "e2_localization")]
    if calibrated_rows:
        grouped_calibrated = _group_rows(calibrated_rows, "localizer_name")
        for name in sorted(grouped_calibrated):
            if name == "transparent_global_threshold":
                continue
            variants.append(
                (
                    _display_name(name),
                    list(grouped_calibrated[name]),
                    "e2_localization_calibrated",
                )
            )
    for variant_name, selected_rows, experiment_id in variants:
        row = [variant_name]
        for group_name, selected in [
            ("All", list(selected_rows)),
            *_stage_groups(selected_rows),
        ]:
            ci = accuracy_ci(
                selected,
                correct_key="correct",
                seed=seed,
                samples=bootstrap_samples,
            )
            row.append(_ci_text(ci.estimate, ci.lower, ci.upper))
            cells.append(
                _cell(
                    "localization",
                    f"{variant_name} / {group_name}",
                    "accuracy",
                    manifests[experiment_id],
                )
            )
        table_rows.append([*row, _short_runs([manifests[experiment_id].run_id])])
    provenance_artifacts = [manifests["e2_localization"]]
    if calibrated_rows:
        provenance_artifacts.append(manifests["e2_localization_calibrated"])
    provenance = _provenance(provenance_artifacts, cells)
    return (
        "tab_localization.tex",
        [
            "Localizer",
            "All",
            "Retrieval",
            "Prompt",
            "Generation",
            "No Fault",
            "Run IDs",
        ],
        table_rows,
        provenance,
    )


def ablation_table(
    rows: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, ResultArtifact],
    seed: int,
    bootstrap_samples: int,
) -> tuple[str, list[str], list[list[str]], dict[str, Any]]:
    """Build the oracle ablation table."""

    grouped = _group_rows(rows, "ablation_name")
    table_rows: list[list[str]] = []
    cells: list[dict[str, str]] = []
    for name in sorted(grouped):
        selected = grouped[name]
        ci = accuracy_ci(
            selected,
            correct_key="correct",
            seed=seed,
            samples=bootstrap_samples,
        )
        oracle_names = _joined_unique(selected, "oracles")
        table_rows.append(
            [
                _display_name(name),
                oracle_names,
                str(len(selected)),
                _ci_text(ci.estimate, ci.lower, ci.upper),
            ]
        )
        cells.append(
            _cell("oracle_ablation", name, "accuracy", manifests["e3_ablation"])
        )
    provenance = _provenance([manifests["e3_ablation"]], cells)
    return (
        "tab_oracle_ablation.tex",
        ["Variant", "Oracles", "N", "Accuracy (95\\% CI)"],
        table_rows,
        provenance,
    )


def mutation_discriminativeness_table(
    rows: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, ResultArtifact],
    seed: int,
    bootstrap_samples: int,
) -> tuple[str, list[str], list[list[str]], dict[str, Any]]:
    """Build a mutation-delta discriminativeness table."""

    selected_rows = _primary_operator_rows(rows)
    flattened = _flatten_operator_deltas(selected_rows)
    grouped = _group_rows(flattened, "operator")
    table_rows: list[list[str]] = []
    cells: list[dict[str, str]] = []
    for operator in sorted(grouped):
        selected = grouped[operator]
        applied = [row for row in selected if row.get("applied") is not False]
        deltas = [float(row["delta"]) for row in applied]
        reject_rate = 1.0 - (len(applied) / len(selected)) if selected else 0.0
        mean_delta = (
            "n/a"
            if not deltas
            else _ci_text_points(
                bootstrap_ci(
                    deltas,
                    lambda sample: mean(float(value) for value in sample),
                    seed=seed,
                    samples=bootstrap_samples,
                )
            )
        )
        table_rows.append(
            [
                operator,
                MUTATION_STAGES.get(operator, "unknown"),
                str(len(applied)),
                _pct(reject_rate),
                mean_delta,
            ]
        )
        cells.append(
            _cell(
                "mutation_discriminativeness",
                operator,
                "mean_delta",
                manifests["e4_separability"],
            )
        )
    provenance = _provenance([manifests["e4_separability"]], cells)
    return (
        "tab_mutation_discriminativeness.tex",
        [
            "Operator",
            "Stage",
            "Applied N",
            "Rejected \\%",
            "$\\Delta\\Omega$ Mean (95\\% CI)",
        ],
        table_rows,
        provenance,
    )


def latency_cost_table(
    rows: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, ResultArtifact],
    seed: int,
    bootstrap_samples: int,
) -> tuple[str, list[str], list[list[str]], dict[str, Any]]:
    """Build the latency and cost table."""

    grouped = _group_rows(rows, "workflow")
    table_rows: list[list[str]] = []
    cells: list[dict[str, str]] = []
    for workflow in sorted(grouped):
        selected = grouped[workflow]
        latency_ci = bootstrap_ci(
            list(selected),
            lambda sample: mean(
                float(row.get("latency_seconds", 0.0)) for row in sample
            ),
            seed=seed,
            samples=bootstrap_samples,
        )
        cost_ci = bootstrap_ci(
            list(selected),
            lambda sample: mean(float(row.get("cost_usd", 0.0)) for row in sample),
            seed=seed,
            samples=bootstrap_samples,
        )
        overhead = mean(float(row.get("overhead_vs_rag", 0.0)) for row in selected)
        table_rows.append(
            [
                _display_name(workflow),
                str(len(selected)),
                _ci_seconds(latency_ci.estimate, latency_ci.lower, latency_ci.upper),
                _ci_usd(cost_ci.estimate, cost_ci.lower, cost_ci.upper),
                _number(overhead, digits=2),
            ]
        )
        cells.append(
            _cell("latency_cost", workflow, "latency_cost", manifests["e5_latency"])
        )
    provenance = _provenance([manifests["e5_latency"]], cells)
    return (
        "tab_latency_cost.tex",
        [
            "Workflow",
            "N",
            "Latency Sec. (95\\% CI)",
            "Cost USD (95\\% CI)",
            "Overhead",
        ],
        table_rows,
        provenance,
    )


def aggregation_table(
    rows: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, ResultArtifact],
    seed: int,
    bootstrap_samples: int,
) -> tuple[str, list[str], list[list[str]], dict[str, Any]]:
    """Build weighted/uniform/confidence-gated aggregation comparison."""

    grouped = _group_rows(rows, "ablation_name")
    table_rows: list[list[str]] = []
    cells: list[dict[str, str]] = []
    for name in sorted(grouped):
        selected = grouped[name]
        ci = accuracy_ci(
            selected,
            correct_key="correct",
            seed=seed,
            samples=bootstrap_samples,
        )
        aggregation = str(selected[0].get("aggregation", name))
        table_rows.append(
            [
                _display_name(name),
                _display_name(aggregation),
                str(len(selected)),
                _ci_text(ci.estimate, ci.lower, ci.upper),
            ]
        )
        cells.append(_cell("aggregation", name, "accuracy", manifests["e6_weighted"]))
    provenance = _provenance([manifests["e6_weighted"]], cells)
    return (
        "tab_aggregation.tex",
        ["Variant", "Aggregator", "N", "Accuracy (95\\% CI)"],
        table_rows,
        provenance,
    )


def write_latex_table(
    path: Path,
    *,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    provenance: Mapping[str, Any],
) -> None:
    """Write a standalone LaTeX tabular fragment with run provenance comments."""

    align = "l" + "r" * (len(headers) - 1)
    lines = [
        "% Generated by experiments/analyze_results.py; do not edit metrics by hand.",
        f"% Run IDs: {', '.join(provenance['run_ids'])}",
        f"% Source files: {', '.join(provenance['source_files'])}",
        f"\\begin{{tabular}}{{{align}}}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(value) for value in row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def architecture_figure() -> tuple[str, str, dict[str, Any]]:
    """Return a clean architecture SVG source."""

    labels = [
        ("RAG", "query, context, answer"),
        ("Mutate", "retrieval, prompt, generation"),
        ("Score", "NLI, semantic, judge"),
        ("Delta", "operator vector"),
        ("Localize", "gate + calibrated classifier"),
    ]
    box_width = 76
    gap = 10
    x0 = 18
    y = 62
    parts = [
        _defs(),
        '<text x="18" y="28" font-size="17" font-weight="700" '
        'font-family="Arial, sans-serif">MutOracle-RAG Pipeline</text>',
    ]
    for index, (title, subtitle) in enumerate(labels):
        x = x0 + index * (box_width + gap)
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="{box_width}" height="58" rx="6" '
                'fill="#f8fafc" stroke="#334155" stroke-width="1.2"/>',
                f'<text x="{x + box_width / 2}" y="{y + 23}" text-anchor="middle" '
                'font-size="12" font-weight="700" font-family="Arial, sans-serif">'
                f"{html.escape(title)}</text>",
                f'<text x="{x + box_width / 2}" y="{y + 42}" text-anchor="middle" '
                'font-size="8" font-family="Arial, sans-serif">'
                f"{html.escape(subtitle)}</text>",
            ]
        )
        if index < len(labels) - 1:
            arrow_x = x + box_width
            parts.append(
                f'<path d="M{arrow_x + 2} {y + 29} L{arrow_x + gap - 2} {y + 29}" '
                'stroke="#0f766e" stroke-width="1.5" marker-end="url(#arrow)"/>'
            )
    svg = _svg(450, 150, parts)
    return "fig_architecture.svg", svg, {"run_ids": [], "source_files": []}


def experiment_design_figure() -> tuple[str, str, dict[str, Any]]:
    """Return an experiment-suite diagram."""

    rows = [
        ("E1", "Response detection", "RAGAS / MetaRAG / MutOracle"),
        ("E2", "Stage localization", "Transparent vs calibrated"),
        ("E3", "Oracle ablation", "NLI, semantic, judge subsets"),
        ("E4", "Mutation signal", "Applied deltas and rejection rates"),
        ("E5", "Audit metadata", "latency, tokens, cache, cost"),
        ("E6", "Aggregation", "uniform, weighted, gated"),
    ]
    parts = [
        '<text x="18" y="28" font-size="17" font-weight="700" '
        'font-family="Arial, sans-serif">Experiment Suite</text>',
    ]
    for index, (eid, title, detail) in enumerate(rows):
        y = 48 + index * 36
        fill = "#ecfdf5" if index % 2 == 0 else "#f8fafc"
        parts.extend(
            [
                f'<rect x="18" y="{y}" width="414" height="28" rx="5" '
                f'fill="{fill}" stroke="#cbd5e1"/>',
                f'<text x="32" y="{y + 19}" font-size="11" font-weight="700" '
                'font-family="Arial, sans-serif">'
                f"{eid}</text>",
                f'<text x="78" y="{y + 19}" font-size="10" '
                'font-family="Arial, sans-serif">'
                f"{html.escape(title)}</text>",
                f'<text x="230" y="{y + 19}" font-size="9" '
                'font-family="Arial, sans-serif" fill="#334155">'
                f"{html.escape(detail)}</text>",
            ]
        )
    return "fig_experiment_design.svg", _svg(450, 285, parts), {
        "run_ids": [],
        "source_files": [],
    }


def calibration_flow_figure() -> tuple[str, str, dict[str, Any]]:
    """Return a validation/test calibration flow diagram."""

    parts = [
        _defs(),
        '<text x="18" y="28" font-size="17" font-weight="700" '
        'font-family="Arial, sans-serif">Validation-Calibrated Localizer</text>',
    ]
    boxes = [
        (20, 58, "FITS validation", "fit gate, scaler, classifier"),
        (236, 58, "FITS test", "held-out evaluation only"),
        (20, 142, "Full delta vector", "11 operator deltas"),
        (236, 142, "Prediction", "retrieval / prompt / generation / no fault"),
    ]
    for x, y, title, subtitle in boxes:
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="176" height="54" rx="6" '
                'fill="#f8fafc" stroke="#334155" stroke-width="1.1"/>',
                f'<text x="{x + 88}" y="{y + 22}" text-anchor="middle" '
                'font-size="11" font-weight="700" font-family="Arial, sans-serif">'
                f"{html.escape(title)}</text>",
                f'<text x="{x + 88}" y="{y + 40}" text-anchor="middle" '
                'font-size="8" font-family="Arial, sans-serif">'
                f"{html.escape(subtitle)}</text>",
            ]
        )
    parts.extend(
        [
            '<path d="M108 112 L108 136" stroke="#0f766e" stroke-width="1.5" '
            'marker-end="url(#arrow)"/>',
            '<path d="M196 169 L230 169" stroke="#0f766e" stroke-width="1.5" '
            'marker-end="url(#arrow)"/>',
            '<path d="M324 112 L324 136" stroke="#0f766e" stroke-width="1.5" '
            'marker-end="url(#arrow)"/>',
            '<path d="M196 85 C214 85 218 85 230 85" stroke="#64748b" '
            'stroke-width="1.2" stroke-dasharray="4 3" marker-end="url(#arrow)"/>',
        ]
    )
    return "fig_calibration_flow.svg", _svg(450, 230, parts), {
        "run_ids": [],
        "source_files": [],
    }


def localizer_accuracy_figure(
    rows: Sequence[Mapping[str, Any]],
    calibrated_rows: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, ResultArtifact],
) -> tuple[str, str, dict[str, Any]]:
    """Return a compact localizer accuracy bar chart."""

    variants: list[tuple[str, Sequence[Mapping[str, Any]], ResultArtifact]] = [
        ("Transparent", rows, manifests["e2_localization"]),
    ]
    grouped = _group_rows(calibrated_rows, "localizer_name")
    for name in [
        "no_fault_gated_delta",
        "stage_threshold_delta",
        "centroid_full_delta",
        "logistic_full_delta",
    ]:
        if name in grouped:
            variants.append(
                (
                    _display_name(name),
                    grouped[name],
                    manifests["e2_localization_calibrated"],
                )
            )
    values = [
        mean(1.0 if row.get("correct") else 0.0 for row in selected)
        for _, selected, _ in variants
    ]
    max_value = max(values or [1.0]) or 1.0
    parts = [
        '<text x="18" y="28" font-size="17" font-weight="700" '
        'font-family="Arial, sans-serif">Localizer Accuracy</text>',
    ]
    for index, ((label, _, _), value) in enumerate(zip(variants, values, strict=True)):
        y = 52 + index * 34
        width = 220 * (value / max_value)
        parts.extend(
            [
                f'<text x="18" y="{y + 15}" font-size="9" '
                'font-family="Arial, sans-serif">'
                f"{html.escape(label)}</text>",
                f'<rect x="166" y="{y}" width="{width:.1f}" height="18" rx="3" '
                'fill="#0f766e"/>',
                f'<text x="{166 + width + 8:.1f}" y="{y + 14}" font-size="10" '
                'font-family="Arial, sans-serif">'
                f"{100 * value:.1f}%</text>",
            ]
        )
    artifacts = [artifact for _, _, artifact in variants]
    return "fig_localizer_accuracy.svg", _svg(450, 245, parts), _provenance(
        artifacts,
        [],
    )


def delta_heatmap(
    rows: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, ResultArtifact],
) -> tuple[str, str, dict[str, Any]]:
    """Return mutation delta heatmap SVG."""

    flattened = _flatten_operator_deltas(_primary_operator_rows(rows))
    values: dict[tuple[str, str], float] = {}
    for stage in STAGE_LABELS:
        stage_rows = [row for row in flattened if row.get("expected_stage") == stage]
        for operator in MUTATION_STAGES:
            selected = [
                float(row["delta"])
                for row in stage_rows
                if row.get("operator") == operator
                and row.get("applied") is not False
            ]
            values[(stage, operator)] = mean(selected)
    svg = _heatmap_svg(
        title="",
        row_labels=list(STAGE_LABELS),
        col_labels=list(MUTATION_STAGES),
        values=values,
        value_format=lambda value: _number(value, digits=2),
        cell_size=30,
        left_margin=78,
        top_margin=58,
        label_font_size=8,
        value_font_size=8,
    )
    artifact = manifests["e4_separability"]
    return "fig_delta_heatmap.svg", svg, _provenance([artifact], [])


def weight_sensitivity_heatmap(
    rows: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, ResultArtifact],
) -> tuple[str, str, dict[str, Any]]:
    """Return aggregation sensitivity SVG."""

    grouped = _group_rows(rows, "ablation_name")
    row_labels = sorted(grouped)
    col_labels = ["accuracy", "mean confidence"]
    values: dict[tuple[str, str], float] = {}
    for name, selected in grouped.items():
        values[(name, "accuracy")] = mean(
            1.0 if row.get("correct") else 0.0 for row in selected
        )
        values[(name, "mean confidence")] = mean(
            float(row.get("confidence", 0.0)) for row in selected
        )
    svg = _heatmap_svg(
        title="Aggregation Sensitivity",
        row_labels=row_labels,
        col_labels=col_labels,
        values=values,
        value_format=lambda value: _pct(value),
        cell_size=68,
        left_margin=126,
        top_margin=72,
        title_font_size=18,
        label_font_size=11,
        value_font_size=11,
    )
    artifact = manifests["e6_weighted"]
    return "fig_weight_sensitivity_heatmap.svg", svg, _provenance([artifact], [])


def confusion_matrix_figure(
    rows: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, ResultArtifact],
) -> tuple[str, str, dict[str, Any]]:
    """Return FITS stage confusion matrix SVG."""

    matrix = confusion_matrix(
        rows,
        expected_key="expected_stage",
        predicted_key="predicted_stage",
        labels=STAGE_LABELS,
    )
    values = {
        (expected, predicted): float(matrix[row_index][col_index])
        for row_index, expected in enumerate(STAGE_LABELS)
        for col_index, predicted in enumerate(STAGE_LABELS)
    }
    svg = _heatmap_svg(
        title="",
        row_labels=list(STAGE_LABELS),
        col_labels=list(STAGE_LABELS),
        values=values,
        value_format=lambda value: str(int(value)),
        cell_size=38,
        left_margin=94,
        top_margin=58,
        label_font_size=8,
        value_font_size=8,
    )
    artifact = manifests.get("e2_localization_calibrated", manifests["e2_localization"])
    return "fig_confusion_matrix.svg", svg, _provenance([artifact], [])


def latency_cost_figure(
    rows: Sequence[Mapping[str, Any]],
    manifests: Mapping[str, ResultArtifact],
) -> tuple[str, str, dict[str, Any]]:
    """Return latency/cost bar chart SVG."""

    grouped = _group_rows(rows, "workflow")
    labels = sorted(grouped)
    latencies = [
        mean(float(row.get("latency_seconds", 0.0)) for row in grouped[label])
        for label in labels
    ]
    costs = [
        mean(float(row.get("cost_usd", 0.0)) for row in grouped[label])
        for label in labels
    ]
    max_latency = max(latencies or [1.0]) or 1.0
    max_cost = max(costs or [1.0]) or 1.0
    bar_max_width = 160
    label_x = 18
    bars_x = 138
    value_x = 312
    row_top = 58
    row_gap = 64
    parts = [
        _defs(),
        '<text x="18" y="28" font-size="18" font-weight="700" '
        'font-family="Arial, sans-serif">Latency and Cost by Workflow</text>',
    ]
    for index, label in enumerate(labels):
        y = row_top + index * row_gap
        latency_width = bar_max_width * (latencies[index] / max_latency)
        cost_width = bar_max_width * (costs[index] / max_cost)
        parts.extend(
            [
                f'<text x="{label_x}" y="{y + 14}" font-size="11" '
                'font-family="Arial, sans-serif">'
                f"{html.escape(_display_name(label))}</text>",
                f'<rect x="{bars_x}" y="{y}" width="{latency_width:.1f}" height="16" '
                'fill="#0f766e"/>',
                f'<rect x="{bars_x}" y="{y + 20}" width="{cost_width:.1f}" height="16" '
                'fill="#64748b"/>',
                f'<text x="{value_x}" y="{y + 14}" font-size="11" '
                f'font-family="Arial, sans-serif">{latencies[index]:.4f}s</text>',
                f'<text x="{value_x}" y="{y + 34}" font-size="11" '
                f'font-family="Arial, sans-serif">${costs[index]:.4f}</text>',
            ]
        )
    svg = _svg(430, max(170, row_top + len(labels) * row_gap + 20), parts)
    artifact = manifests["e5_latency"]
    return "fig_latency_cost.svg", svg, _provenance([artifact], [])


def write_traceability_report(
    path: Path,
    *,
    entries: Sequence[Mapping[str, Any]],
    artifacts: Sequence[ResultArtifact],
    mode: str,
) -> None:
    """Write the table/figure to run-ID traceability report."""

    lines = [
        "# Phase 9 Traceability Report",
        "",
        f"Generated from `{mode}` result artifacts. Metrics are derived from saved "
        "JSONL files through `experiments/analyze_results.py`.",
        "",
        "## Source Runs",
        "",
        "| Experiment | Run ID | Git Commit | Raw JSONL | Manifest |",
        "| --- | --- | --- | --- | --- |",
    ]
    for artifact in artifacts:
        lines.append(
            "| "
            f"{artifact.experiment_id} | {artifact.run_id} | "
            f"{artifact.git_commit[:12]} | {_path_text(artifact.raw_jsonl)} | "
            f"{_path_text(artifact.manifest_json)} |"
        )
    lines.extend(["", "## Asset Mapping", ""])
    for entry in entries:
        lines.append(f"### `{entry['asset']}`")
        lines.append("")
        run_ids = ", ".join(entry.get("run_ids", [])) or "analysis diagram"
        lines.append(f"- Type: {entry['type']}")
        lines.append(f"- Run IDs: {run_ids}")
        lines.append(
            f"- Source files: {', '.join(entry.get('source_files', [])) or 'n/a'}"
        )
        cells = entry.get("cells", [])
        if cells:
            lines.append("- Cell provenance:")
            for cell in cells:
                lines.append(
                    "  - "
                    f"{cell['table']} / {cell['row']} / {cell['metric']} -> "
                    f"{cell['run_id']}"
                )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("experiments/results"))
    parser.add_argument("--paper-dir", type=Path, default=Path("paper"))
    parser.add_argument("--mode", choices=["smoke", "dev", "full"], default="smoke")
    parser.add_argument("--tables", action="store_true", help="Generate tables only.")
    parser.add_argument("--figures", action="store_true", help="Generate figures only.")
    parser.add_argument("--duckdb-path", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--require-experiments",
        nargs="+",
        default=list(REQUIRED_EXPERIMENTS),
        help="Required experiment IDs; missing artifacts fail clearly.",
    )
    return parser.parse_args()


def _rows_by_experiment(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("experiment_id", ""))].append(row)
    return grouped


def _ensure_required_experiment_rows(
    rows_by_experiment: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    required_experiments: Sequence[str],
) -> None:
    missing = [
        experiment_id
        for experiment_id in required_experiments
        if not rows_by_experiment.get(experiment_id)
    ]
    if not missing:
        return
    available = (
        ", ".join(sorted(name for name, rows in rows_by_experiment.items() if rows))
        or "none"
    )
    msg = (
        "Imported raw results are missing rows for required experiments: "
        f"{', '.join(missing)}. Available experiment_id values: {available}. "
        "Check that each raw JSONL record includes the expected experiment_id."
    )
    raise AnalysisError(msg)


def _fetch_all_dicts(
    connection: duckdb.DuckDBPyConnection,
    query: str,
) -> list[dict[str, Any]]:
    columns = [column[0] for column in connection.execute(query).description]
    return [dict(zip(columns, row, strict=True)) for row in connection.fetchall()]


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return grouped


def _stage_groups(
    rows: Sequence[Mapping[str, Any]],
) -> list[tuple[str, list[Mapping[str, Any]]]]:
    grouped = _group_rows(rows, "expected_stage")
    return [
        (_display_name(stage), grouped[stage])
        for stage in STAGE_LABELS
        if stage in grouped
    ]


def _flatten_operator_deltas(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        deltas = row.get("operator_deltas", {})
        if not isinstance(deltas, dict):
            continue
        status = row.get("operator_status", {})
        status = status if isinstance(status, dict) else {}
        for operator, delta in deltas.items():
            if delta is None:
                continue
            operator_status = status.get(operator, {})
            operator_status = (
                operator_status if isinstance(operator_status, dict) else {}
            )
            flattened.append(
                {
                    "operator": str(operator),
                    "delta": float(delta),
                    "applied": operator_status.get("applied"),
                    "rejected": operator_status.get("rejected"),
                    "expected_stage": row.get("expected_stage"),
                    "qid": row.get("qid"),
                    "seed": row.get("seed"),
                }
            )
    return flattened


def _primary_operator_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return all-operator rows when E4 contains operator-drop variants."""

    selected = [
        row for row in rows if str(row.get("ablation_name", "")) == "all_operators"
    ]
    return selected or list(rows)


def _joined_unique(rows: Sequence[Mapping[str, Any]], key: str) -> str:
    values: list[str] = []
    for row in rows:
        raw = row.get(key, [])
        if isinstance(raw, list):
            values.extend(str(item) for item in raw)
        elif raw:
            values.append(str(raw))
    return ", ".join(sorted(set(values))) or "n/a"


def _provenance(
    artifacts: Sequence[ResultArtifact],
    cells: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    return {
        "run_ids": [artifact.run_id for artifact in artifacts],
        "source_files": [_path_text(artifact.raw_jsonl) for artifact in artifacts],
        "cells": [dict(cell) for cell in cells],
    }


def _cell(
    table: str,
    row: str,
    metric: str,
    artifact: ResultArtifact,
) -> dict[str, str]:
    return {
        "table": table,
        "row": row,
        "metric": metric,
        "run_id": artifact.run_id,
        "source_file": _path_text(artifact.raw_jsonl),
    }


def _display_name(value: str) -> str:
    normalized = value.strip().lower()
    canonical = {
        "ci": "CI",
        "cr": "CR",
        "cs": "CS",
        "qp": "QP",
        "qn": "QN",
        "qd": "QD",
        "qi": "QI",
        "fs": "FS",
        "fa": "FA",
        "fe": "FE",
        "gn": "GN",
        "no_fault": "No Fault",
        "no_fault_detected": "No Fault Detected",
    }
    if normalized in canonical:
        return canonical[normalized]
    rendered = value.replace("_", " ").title()
    return (
        rendered.replace("Nli", "NLI")
        .replace("Llm", "LLM")
        .replace("Ragas", "RAGAS")
        .replace("Metarag", "MetaRAG")
        .replace("Mutoracle", "MutOracle")
    )


def _axis_label(value: str) -> str:
    if value.strip().lower() in {"no_fault", "no_fault_detected"}:
        return "No Fault"
    return _display_name(value)


def _short_runs(run_ids: Sequence[str]) -> str:
    return ", ".join(run_id[:8] for run_id in run_ids if run_id)


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def _number(value: float, *, digits: int = 3) -> str:
    if math.isclose(value, 0.0, abs_tol=10 ** (-(digits + 1))):
        value = 0.0
    return f"{value:.{digits}f}"


def _ci_text(estimate: float, lower: float, upper: float) -> str:
    return f"{_pct(estimate)} [{_pct(lower)}, {_pct(upper)}]"


def _ci_text_points(ci: Any) -> str:
    estimate = _number(float(ci.estimate), digits=1)
    lower = _number(float(ci.lower), digits=1)
    upper = _number(float(ci.upper), digits=1)
    return f"{estimate} [{lower}, {upper}]"


def _ci_seconds(estimate: float, lower: float, upper: float) -> str:
    return f"{estimate:.4f} [{lower:.4f}, {upper:.4f}]"


def _ci_usd(estimate: float, lower: float, upper: float) -> str:
    return f"{estimate:.4f} [{lower:.4f}, {upper:.4f}]"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _path_text(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def _svg(width: int, height: int, parts: Iterable[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        + "\n".join(parts)
        + "\n</svg>\n"
    )


def _defs() -> str:
    return (
        "<defs>"
        '<marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" '
        'orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#0f766e"/>'
        "</marker>"
        "</defs>"
    )


def _heatmap_svg(
    *,
    title: str,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    values: Mapping[tuple[str, str], float],
    value_format: Any,
    cell_size: int = 56,
    left_margin: int = 120,
    top_margin: int = 74,
    title_font_size: int = 18,
    label_font_size: int = 11,
    value_font_size: int = 11,
) -> str:
    cell = cell_size
    left = left_margin
    top = top_margin
    width = left + cell * len(col_labels) + 22
    height = top + cell * len(row_labels) + 28
    max_value = max([abs(value) for value in values.values()] + [1e-9])
    parts = []
    if title:
        parts.append(
            f'<text x="18" y="28" font-size="{title_font_size}" '
            'font-weight="700" font-family="Arial, sans-serif">'
            f"{html.escape(title)}</text>"
        )
    col_label_y = max(18, top - 14)
    for col_index, label in enumerate(col_labels):
        x = left + col_index * cell + cell / 2
        parts.append(
            f'<text x="{x:.1f}" y="{col_label_y}" text-anchor="middle" '
            f'font-size="{label_font_size}" '
            'font-family="Arial, sans-serif">'
            f"{html.escape(_axis_label(label))}</text>"
        )
    for row_index, row_label in enumerate(row_labels):
        y = top + row_index * cell
        parts.append(
            f'<text x="{left - 10}" y="{y + cell / 2 + 4:.1f}" '
            f'text-anchor="end" font-size="{label_font_size}" '
            'font-family="Arial, sans-serif">'
            f"{html.escape(_axis_label(row_label))}</text>"
        )
        for col_index, col_label in enumerate(col_labels):
            x = left + col_index * cell
            value = float(values.get((row_label, col_label), 0.0))
            intensity = min(1.0, abs(value) / max_value)
            fill = _teal_scale(intensity)
            parts.extend(
                [
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                    f'fill="{fill}" stroke="#ffffff" stroke-width="0.8"/>',
                    f'<text x="{x + cell / 2:.1f}" y="{y + cell / 2 + 4:.1f}" '
                    f'text-anchor="middle" font-size="{value_font_size}" '
                    'font-family="Arial, sans-serif" '
                    'fill="#0f172a">'
                    f"{html.escape(value_format(value))}</text>",
                ]
            )
    return _svg(width, height, parts)


def _teal_scale(intensity: float) -> str:
    base = (15, 118, 110)
    light = (236, 253, 245)
    rgb = tuple(
        int(light[index] + (base[index] - light[index]) * intensity)
        for index in range(3)
    )
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


if __name__ == "__main__":
    main()
