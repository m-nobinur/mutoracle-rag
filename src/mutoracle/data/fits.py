"""FITS v1.0.0 builder and validation logic."""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from statistics import fmean
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from mutoracle.contracts import RAGRun
from mutoracle.data.loaders import (
    SourceExample,
    build_noise_pool,
    dataset_manifests,
    load_rgb_subset,
    load_triviaqa_subset,
)
from mutoracle.data.manifest import json_dump, sha256_file
from mutoracle.mutations.prompt import QueryNegationMutation

FaultLabel = Literal["retrieval", "prompt", "generation", "no_fault"]
SplitName = Literal["validation", "test"]

LABELS: tuple[FaultLabel, ...] = ("retrieval", "prompt", "generation", "no_fault")
QUERY_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?")


class FITSRecord(BaseModel):
    """One FITS JSONL example."""

    model_config = ConfigDict(extra="forbid")

    qid: str
    query: str
    gt_answer: str
    fault_stage: FaultLabel
    injection: dict[str, Any]
    source: Literal["rgb", "triviaqa"]
    source_qid: str
    split: SplitName
    build_seed: int


class FITSValidationReport(BaseModel):
    """Quality-gate report for a FITS build."""

    model_config = ConfigDict(extra="forbid")

    total_examples: int
    labels: dict[str, int]
    splits: dict[str, int]
    stratification_within_tolerance: bool
    qid_overlap_count: int
    deterministic_checksum: str
    audit_sample_size: int
    audit_label_correctness: float
    source_query_length_mean: float
    fits_query_length_mean: float
    query_length_within_tolerance: bool
    passed: bool


def build_fits_dataset(
    *,
    output_root: Path = Path("data"),
    seed: int = 2026,
    version: str = "fits_v1.0.0",
    examples_per_label: int = 75,
    validation_per_label: int = 15,
    audit_sample_size: int = 50,
    build_date: str | None = None,
    force_rebuild: bool = False,
    query_length_tolerance: float = 0.20,
) -> dict[str, Path]:
    """Build manifests, FITS JSONL splits, and validation reports."""

    rng = Random(seed)
    fits_root = output_root / "fits"
    version_root = fits_root / version
    paths = _artifact_paths(output_root=output_root, version_root=version_root)

    if _has_complete_artifact(paths) and not force_rebuild:
        return paths
    if version_root.exists() and not force_rebuild:
        msg = (
            f"FITS artifact directory already exists: {version_root}. "
            "Use force_rebuild=True to rebuild this frozen version."
        )
        raise FileExistsError(msg)

    version_root.mkdir(parents=True, exist_ok=True)

    sources = _source_pool(examples_per_label=examples_per_label)
    noise = build_noise_pool(limit=examples_per_label * len(LABELS))
    records = _make_records(
        sources=sources,
        noise=noise,
        seed=seed,
        rng=rng,
        examples_per_label=examples_per_label,
        validation_per_label=validation_per_label,
    )

    all_path = paths["fits"]
    validation_path = paths["validation"]
    test_path = paths["test"]
    _write_jsonl(all_path, records)
    _write_jsonl(
        validation_path,
        [record for record in records if record.split == "validation"],
    )
    _write_jsonl(test_path, [record for record in records if record.split == "test"])

    audit_records = _audit_sample(records, seed=seed, size=audit_sample_size)
    audit_path = paths["audit_sample"]
    _write_jsonl(audit_path, audit_records)
    audit_label_correctness = _audit_label_correctness(audit_records)
    source_query_length_mean = _mean_query_length(item.query for item in sources)
    fits_query_length_mean = _mean_query_length(item.query for item in records)

    report = validate_fits_records(
        records,
        expected_per_label=examples_per_label,
        tolerance=0.05,
        audit_sample_size=len(audit_records),
        audit_label_correctness=audit_label_correctness,
        source_query_length_mean=source_query_length_mean,
        fits_query_length_mean=fits_query_length_mean,
        query_length_tolerance=query_length_tolerance,
        checksum=sha256_file(all_path),
    )
    report_path = paths["quality_report"]
    json_dump(report.model_dump(mode="json"), report_path)

    manifest_date = build_date or datetime.now(UTC).date().isoformat()
    source_manifests = dataset_manifests(build_date=manifest_date)
    manifest = {
        "artifact": version,
        "build_seed": seed,
        "created_at": f"{manifest_date}T00:00:00+00:00",
        "code_commit": _git_commit_hash(),
        "source_revisions": [item.model_dump(mode="json") for item in source_manifests],
        "files": {
            "fits_jsonl": {
                "path": str(all_path),
                "checksum": sha256_file(all_path),
            },
            "validation_jsonl": {
                "path": str(validation_path),
                "checksum": sha256_file(validation_path),
            },
            "test_jsonl": {
                "path": str(test_path),
                "checksum": sha256_file(test_path),
            },
            "audit_sample_jsonl": {
                "path": str(audit_path),
                "checksum": sha256_file(audit_path),
            },
        },
        "quality_gates": report.model_dump(mode="json"),
    }
    manifest_path = paths["manifest"]
    json_dump(manifest, manifest_path)
    json_dump(
        {"datasets": [item.model_dump(mode="json") for item in source_manifests]},
        output_root / "manifests" / "datasets.json",
    )
    _write_dataset_card(paths["dataset_card"], report=report)

    return paths


def validate_fits_records(
    records: list[FITSRecord],
    *,
    expected_per_label: int,
    tolerance: float,
    audit_sample_size: int,
    audit_label_correctness: float,
    source_query_length_mean: float,
    fits_query_length_mean: float,
    query_length_tolerance: float,
    checksum: str,
) -> FITSValidationReport:
    """Validate FITS stratification, split integrity, and audit threshold."""

    label_counts = Counter(record.fault_stage for record in records)
    split_counts = Counter(record.split for record in records)
    total = len(records)
    expected_share = 1.0 / len(LABELS)
    stratified = all(
        abs((label_counts[label] / total) - expected_share) <= tolerance
        and label_counts[label] == expected_per_label
        for label in LABELS
    )

    validation_qids = {record.qid for record in records if record.split == "validation"}
    test_qids = {record.qid for record in records if record.split == "test"}
    overlap = validation_qids & test_qids
    query_length_within_tolerance = (
        _relative_mean_delta(source_query_length_mean, fits_query_length_mean)
        <= query_length_tolerance
    )
    passed = (
        stratified
        and not overlap
        and audit_sample_size >= 50
        and audit_label_correctness >= 0.95
        and query_length_within_tolerance
    )
    return FITSValidationReport(
        total_examples=total,
        labels={label: label_counts[label] for label in LABELS},
        splits={
            "validation": split_counts["validation"],
            "test": split_counts["test"],
        },
        stratification_within_tolerance=stratified,
        qid_overlap_count=len(overlap),
        deterministic_checksum=checksum,
        audit_sample_size=audit_sample_size,
        audit_label_correctness=audit_label_correctness,
        source_query_length_mean=source_query_length_mean,
        fits_query_length_mean=fits_query_length_mean,
        query_length_within_tolerance=query_length_within_tolerance,
        passed=passed,
    )


def _source_pool(*, examples_per_label: int) -> list[SourceExample]:
    trivia = load_triviaqa_subset(limit=examples_per_label * 2)
    rgb = load_rgb_subset(limit=examples_per_label * 2)
    return [item for pair in zip(trivia, rgb, strict=True) for item in pair]


def _make_records(
    *,
    sources: list[SourceExample],
    noise: list[dict[str, str]],
    seed: int,
    rng: Random,
    examples_per_label: int,
    validation_per_label: int,
) -> list[FITSRecord]:
    records: list[FITSRecord] = []
    source_index = 0
    noise_index = 0
    for label in LABELS:
        label_records: list[FITSRecord] = []
        for local_index in range(examples_per_label):
            source = sources[source_index % len(sources)]
            source_index += 1
            noise_doc = noise[noise_index % len(noise)]
            noise_index += 1
            query, injection = _query_and_injection(
                label=label,
                source=source,
                noise_doc=noise_doc,
            )
            split: SplitName = (
                "validation" if local_index < validation_per_label else "test"
            )
            qid = f"fits_{len(records) + len(label_records):06d}"
            label_records.append(
                FITSRecord(
                    qid=qid,
                    query=query,
                    gt_answer=source.gt_answer,
                    fault_stage=label,
                    injection=injection,
                    source=cast("Literal['rgb', 'triviaqa']", source.source),
                    source_qid=source.source_qid,
                    split=split,
                    build_seed=seed,
                )
            )
        rng.shuffle(label_records)
        records.extend(label_records)
    return sorted(records, key=lambda record: record.qid)


def _query_and_injection(
    *,
    label: FaultLabel,
    source: SourceExample,
    noise_doc: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    if label == "prompt":
        negated = _negated_query(source.query)
        if negated is not None:
            return (
                negated,
                {
                    "method": "query_negation",
                    "operator": "QN",
                    "verifier": "grammaticality_audit_sample",
                },
            )
        return (
            _controlled_prompt_perturbation(source.query),
            {
                "method": "controlled_prompt_perturbation",
                "operator": "QN_fallback",
                "verifier": "grammaticality_audit_sample",
                "reason": "query_shape_cannot_be_grammatically_negated",
            },
        )

    return source.query, _injection(
        label=label,
        source=source,
        noise_doc=noise_doc,
    )


def _injection(
    *,
    label: FaultLabel,
    source: SourceExample,
    noise_doc: dict[str, str],
) -> dict[str, Any]:
    if label == "retrieval":
        return {
            "method": "remove_ground_truth_passage",
            "removed_doc_id": source.supporting_doc_id,
            "verifier": "ground_truth_doc_absent_from_top_k",
        }
    if label == "prompt":
        return {
            "method": "query_negation",
            "operator": "QN",
            "verifier": "grammaticality_audit_sample",
        }
    if label == "generation":
        return {
            "method": "distractor_at_top1",
            "distractor_doc_id": source.distractor_doc_id,
            "noise_doc_id": noise_doc["doc_id"],
            "verifier": "distractor_factoid_contradiction",
        }
    return {"method": "none", "verifier": "no_injection"}


def _write_jsonl(path: Path, records: list[FITSRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(record.model_dump(mode="json"), sort_keys=True) for record in records
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _audit_sample(
    records: list[FITSRecord],
    *,
    seed: int,
    size: int,
) -> list[FITSRecord]:
    rng = Random(seed + 1)
    selected = list(records)
    rng.shuffle(selected)
    return sorted(selected[:size], key=lambda record: record.qid)


def _audit_label_correctness(records: list[FITSRecord]) -> float:
    if not records:
        return 0.0

    expected_methods: dict[FaultLabel, set[str]] = {
        "retrieval": {"remove_ground_truth_passage"},
        "prompt": {"query_negation", "controlled_prompt_perturbation"},
        "generation": {"distractor_at_top1"},
        "no_fault": {"none"},
    }
    correct = sum(
        1
        for record in records
        if record.injection.get("method") in expected_methods[record.fault_stage]
    )
    return correct / len(records)


def _write_dataset_card(path: Path, *, report: FITSValidationReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# FITS v1.0.0 Dataset Card",
                "",
                "FITS is the Phase 6 fault-injection test split for MutOracle-RAG.",
                "It contains balanced single-stage labels for retrieval, prompt,",
                "generation, and no-fault controls.",
                "",
                "## Schema",
                "",
                "`qid`, `query`, `gt_answer`, `fault_stage`, `injection`,",
                "`source`, `source_qid`, `split`, and `build_seed`.",
                "",
                "## Quality Gates",
                "",
                f"- Total examples: {report.total_examples}",
                f"- Label counts: {report.labels}",
                f"- Split counts: {report.splits}",
                f"- Validation/test overlap: {report.qid_overlap_count}",
                f"- Audit label correctness: {report.audit_label_correctness:.2%}",
                (
                    "- Query length means (source/fits): "
                    f"{report.source_query_length_mean:.2f}/"
                    f"{report.fits_query_length_mean:.2f}"
                ),
                (f"- Query-length gate passed: {report.query_length_within_tolerance}"),
                f"- Passed: {report.passed}",
                "",
                "Raw versioned JSONL files are generated under `data/fits/` and",
                "are intentionally ignored by git. Commit manifests and reports,",
                "then publish frozen data as release assets when needed.",
                "Rebuilding a frozen version requires `--force`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _artifact_paths(*, output_root: Path, version_root: Path) -> dict[str, Path]:
    return {
        "manifest": output_root / "fits" / "manifest.json",
        "fits": version_root / "fits.jsonl",
        "validation": version_root / "validation.jsonl",
        "test": version_root / "test.jsonl",
        "quality_report": version_root / "quality_gate_report.json",
        "audit_sample": version_root / "audit_sample.jsonl",
        "dataset_card": output_root / "FITS_DATASET.md",
    }


def _has_complete_artifact(paths: dict[str, Path]) -> bool:
    required = (
        "manifest",
        "fits",
        "validation",
        "test",
        "quality_report",
        "audit_sample",
        "dataset_card",
    )
    return all(paths[name].exists() for name in required)


def _negated_query(query: str) -> str | None:
    mutation = QueryNegationMutation()
    mutated = mutation.apply(
        RAGRun(query=query, passages=[], answer="", metadata={}),
        rng=Random(0),
    )
    mutation_record = mutated.metadata.get("mutation", {})
    if isinstance(mutation_record, dict) and mutation_record.get("rejected"):
        return None
    if mutated.query == query:
        return None
    return mutated.query


def _controlled_prompt_perturbation(query: str) -> str:
    stripped = query.strip()
    if stripped.endswith("?"):
        stripped = stripped[:-1]
    return f"Not: {stripped}?"


def _mean_query_length(queries: Iterable[str]) -> float:
    lengths = [len(QUERY_TOKEN_PATTERN.findall(str(query))) for query in queries]
    if not lengths:
        return 0.0
    return float(fmean(lengths))


def _relative_mean_delta(baseline: float, observed: float) -> float:
    if baseline <= 0.0:
        return 0.0 if observed <= 0.0 else 1.0
    return abs(observed - baseline) / baseline


def _git_commit_hash() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return output.strip() or "unknown"
