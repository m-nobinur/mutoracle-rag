"""Shared helpers for Phase 8 config-driven experiment scripts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Literal

import yaml

from mutoracle.aggregation import (
    ConfidenceGatedAggregator,
    UniformAggregator,
    WeightedAggregator,
)
from mutoracle.cache import SQLiteCacheLedger, UsageSummary
from mutoracle.config import MutOracleConfig, load_config
from mutoracle.contracts import Aggregator, DiagnosisStage, RAGRun
from mutoracle.mutations.base import content_similarity
from mutoracle.oracles import LLMJudgeOracle, NLIOracle, SemanticSimilarityOracle
from mutoracle.oracles.base import context_text

RunMode = Literal["smoke", "full"]
OracleMode = Literal["fixture", "real"]

DEFAULT_SEEDS = (13, 42, 91)
DEFAULT_FITS_PATH = Path("data/fits/fits_v1.0.0/fits.jsonl")
POLICY_CONFIRM_COST_CAP_USD = 5.0


@dataclass(frozen=True)
class ExperimentRunSettings:
    """Resolved settings for one smoke or full experiment invocation."""

    experiment_id: str
    title: str
    mode: RunMode
    config_path: Path
    dataset_path: Path
    split: str
    query_limit: int
    seeds: list[int]
    output_dir: Path
    cost_cap_usd: float
    estimated_cost_per_example_usd: float
    require_smoke_before_full: bool


@dataclass(frozen=True)
class ArtifactPaths:
    """Standard Phase 8 result artifact paths."""

    raw_jsonl: Path
    summary_csv: Path
    manifest_json: Path
    failures_jsonl: Path
    config_snapshot_yaml: Path
    duckdb_sql: Path


class FixtureOracle:
    """Credential-free oracle used by Phase 8 smoke runs."""

    def __init__(self, name: str, *, query_weight: float = 0.0) -> None:
        self.name = name
        self._query_weight = query_weight

    def score(self, run: RAGRun) -> float:
        support = content_similarity(context_text(run), run.answer)
        if self._query_weight == 0.0:
            return support
        query_alignment = content_similarity(run.query, run.answer)
        return (
            1.0 - self._query_weight
        ) * support + self._query_weight * query_alignment


class FITSRecordPipeline:
    """Single-record RAG pipeline adapter for FITS experiment examples."""

    def __init__(self, record: Mapping[str, Any], *, seed: int) -> None:
        self._record = dict(record)
        self._seed = seed

    def run(self, query: str) -> RAGRun:
        return rag_run_from_fits_record(self._record, query=query, seed=self._seed)


def load_experiment_config(path: Path) -> dict[str, Any]:
    """Load a Phase 8 YAML config file."""

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, dict):
        msg = f"Experiment config must be a mapping: {path}"
        raise ValueError(msg)
    return raw


def resolve_run_settings(
    config_path: Path,
    *,
    mode: RunMode,
    default_experiment_id: str,
) -> ExperimentRunSettings:
    """Resolve common experiment settings for a run mode."""

    raw = load_experiment_config(config_path)
    experiment = _mapping(raw.get("experiment"))
    mode_config = _mapping(raw.get(mode))
    dataset = _mapping(raw.get("dataset"))
    cost = _mapping(raw.get("cost_gate"))

    experiment_id = str(experiment.get("id", default_experiment_id))
    title = str(experiment.get("title", experiment_id))
    output_dir = Path(
        str(experiment.get("output_dir", f"experiments/results/{experiment_id}"))
    )
    dataset_path = Path(str(dataset.get("path", DEFAULT_FITS_PATH)))
    split = str(dataset.get("split", "test"))
    query_limit = int(mode_config.get("query_limit", 5 if mode == "smoke" else 60))
    if query_limit < 1:
        msg = "query_limit must be at least 1"
        raise ValueError(msg)

    seeds = [int(seed) for seed in mode_config.get("seeds", DEFAULT_SEEDS)]
    if not seeds:
        msg = "At least one seed is required"
        raise ValueError(msg)

    return ExperimentRunSettings(
        experiment_id=experiment_id,
        title=title,
        mode=mode,
        config_path=config_path,
        dataset_path=dataset_path,
        split=split,
        query_limit=query_limit,
        seeds=seeds,
        output_dir=output_dir,
        cost_cap_usd=float(cost.get("max_estimated_cost_usd", 0.0)),
        estimated_cost_per_example_usd=float(
            cost.get("estimated_cost_per_example_usd", 0.0)
        ),
        require_smoke_before_full=bool(cost.get("require_smoke_before_full", True)),
    )


def artifact_paths(settings: ExperimentRunSettings) -> ArtifactPaths:
    """Return standard artifact paths for one experiment invocation."""

    prefix = f"{settings.experiment_id}_{settings.mode}"
    return ArtifactPaths(
        raw_jsonl=settings.output_dir / f"{prefix}_raw.jsonl",
        summary_csv=settings.output_dir / f"{prefix}_summary.csv",
        manifest_json=settings.output_dir / f"{prefix}_manifest.json",
        failures_jsonl=settings.output_dir / f"{prefix}_failures.jsonl",
        config_snapshot_yaml=settings.output_dir / f"{prefix}_config_snapshot.yaml",
        duckdb_sql=settings.output_dir / f"{prefix}_duckdb.sql",
    )


def snapshot_config(settings: ExperimentRunSettings, paths: ArtifactPaths) -> None:
    """Save the exact YAML config beside result files."""

    paths.config_snapshot_yaml.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(settings.config_path, paths.config_snapshot_yaml)


def selected_fits_records(settings: ExperimentRunSettings) -> list[dict[str, Any]]:
    """Load and select experiment records for the configured split and query limit."""

    records: list[dict[str, Any]] = []
    with settings.dataset_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if settings.split != "all" and row.get("split") != settings.split:
                continue
            records.append(row)
            if len(records) >= settings.query_limit:
                break
    if not records:
        msg = (
            f"No experiment records selected from {settings.dataset_path} "
            f"for split={settings.split!r}."
        )
        raise ValueError(msg)
    return records


def expected_diagnosis_stage(record: Mapping[str, Any]) -> DiagnosisStage:
    """Map dataset labels to the localizer diagnosis label set."""

    explicit = str(record.get("expected_stage", "")).strip()
    if explicit:
        if explicit in {"retrieval", "prompt", "generation", "no_fault_detected"}:
            return explicit  # type: ignore[return-value]
        msg = f"Unsupported expected_stage: {explicit}"
        raise ValueError(msg)

    stage = str(record.get("fault_stage", "no_fault"))
    if stage == "no_fault":
        return "no_fault_detected"
    if stage == "no_fault_detected":
        return "no_fault_detected"
    if stage in {"retrieval", "prompt", "generation"}:
        return stage  # type: ignore[return-value]
    msg = f"Unsupported dataset fault_stage: {stage}"
    raise ValueError(msg)


def expected_detection_label(
    record: Mapping[str, Any],
) -> Literal["faithful", "hallucinated"]:
    """Return the response-level target label for baseline experiments."""

    explicit = str(record.get("expected_label", "")).strip().lower()
    if explicit in {"faithful", "hallucinated"}:
        return explicit  # type: ignore[return-value]

    label = str(record.get("label", "")).strip().lower()
    if label in {"faithful", "hallucinated"}:
        return label  # type: ignore[return-value]
    if label in {"supported", "entailment", "correct", "no_fault"}:
        return "faithful"
    if label in {"unsupported", "hallucination", "incorrect", "fault"}:
        return "hallucinated"

    stage = str(record.get("fault_stage", "")).strip()
    return (
        "faithful"
        if stage in {"no_fault", "no_fault_detected", "faithful"}
        else "hallucinated"
    )


def rag_run_from_fits_record(
    record: Mapping[str, Any],
    *,
    query: str | None = None,
    seed: int,
) -> RAGRun:
    """Materialize a dataset record as a shared RAGRun for experiment runners."""

    gt_answer = str(record["gt_answer"])
    stage = str(record.get("fault_stage", "no_fault"))
    run_query = query or str(record["query"])
    support_passage = _support_text(record, gt_answer)
    distractor_passage = _distractor_text(record)
    generated_fault_answer = _fault_answer(record, gt_answer)
    if stage == "retrieval":
        passages = [distractor_passage]
        answer = generated_fault_answer or f"The context does not support {gt_answer}."
    elif stage == "prompt":
        passages = [support_passage]
        answer = generated_fault_answer or f"Not {gt_answer}."
    elif stage == "generation":
        passages = [support_passage]
        answer = generated_fault_answer or f"{gt_answer} is not the correct answer."
    else:
        passages = [support_passage]
        answer = str(record.get("faithful_answer", gt_answer))

    generation_model = str(record.get("generation_model", "fixture-fits-generator"))
    provider_route = str(record.get("provider_route", "fixture"))

    return RAGRun(
        query=run_query,
        passages=passages,
        answer=answer,
        metadata={
            "fits": {
                "qid": record["qid"],
                "fault_stage": stage,
                "source": record.get("source"),
                "source_qid": record.get("source_qid"),
                "split": record.get("split"),
            },
            "generation": {
                "model": generation_model,
                "provider": provider_route,
                "provider_route": provider_route,
                "estimated_cost_usd": 0.0,
                "latency_seconds": 0.0,
                "seed": seed,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            },
        },
    )


def resolve_oracle_mode(
    section: Mapping[str, Any],
    *,
    default: OracleMode = "fixture",
) -> OracleMode:
    """Resolve fixture vs real oracle mode from one config section."""

    value = str(section.get("oracle_mode", default)).strip().lower()
    if value in {"fixture", "real"}:
        return value  # type: ignore[return-value]
    msg = "oracle_mode must be either 'fixture' or 'real'."
    raise ValueError(msg)


def resolve_runtime_config_path(
    raw_config: Mapping[str, Any],
    *,
    section: Mapping[str, Any] | None = None,
) -> Path | None:
    """Return runtime config path from section-level or top-level config fields."""

    source: object | None = None
    if section is not None and "runtime_config" in section:
        source = section.get("runtime_config")
    elif "runtime_config" in raw_config:
        source = raw_config.get("runtime_config")

    if source is None:
        return None
    if isinstance(source, Path):
        return source
    if isinstance(source, str) and source.strip():
        return Path(source.strip())
    msg = "runtime_config must be a path string when provided."
    raise ValueError(msg)


def load_runtime_config(runtime_config_path: Path | None) -> MutOracleConfig:
    """Load MutOracle runtime config, defaulting to project config resolution."""

    return load_config(runtime_config_path)


def real_oracles(
    names: Sequence[str],
    *,
    config: MutOracleConfig,
    ledger: SQLiteCacheLedger,
) -> list[Any]:
    """Build model-backed oracle instances by canonical oracle name."""

    selected = [str(name).strip() for name in names]
    oracles: list[Any] = []
    for name in selected:
        if name == "nli":
            oracles.append(NLIOracle(config=config, ledger=ledger))
        elif name == "semantic_similarity":
            oracles.append(SemanticSimilarityOracle(config=config, ledger=ledger))
        elif name == "llm_judge":
            oracles.append(LLMJudgeOracle(config=config, ledger=ledger))
        else:
            msg = f"Unsupported oracle name: {name}"
            raise ValueError(msg)
    return oracles


def real_model_ids(
    names: Sequence[str],
    *,
    config: MutOracleConfig,
    generation_model: str = "fixture-fits-generator",
) -> list[str]:
    """Return model IDs for real-oracle runs matching selected oracle names."""

    mapping = {
        "nli": config.oracles.nli_model,
        "semantic_similarity": config.oracles.semantic_model,
        "llm_judge": config.models.judge,
    }
    model_ids: list[str] = [generation_model]
    for name in names:
        model_id = mapping.get(str(name).strip())
        if model_id and model_id not in model_ids:
            model_ids.append(model_id)
    return model_ids


def usage_delta(before: UsageSummary, after: UsageSummary) -> dict[str, float | int]:
    """Return non-negative usage deltas between two ledger snapshots."""

    prompt_tokens = max(0, after.prompt_tokens - before.prompt_tokens)
    completion_tokens = max(0, after.completion_tokens - before.completion_tokens)
    total_tokens = prompt_tokens + completion_tokens
    return {
        "requests": max(0, after.requests - before.requests),
        "live_requests": max(0, after.live_requests - before.live_requests),
        "cache_hits": max(0, after.cache_hits - before.cache_hits),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": max(0.0, after.total_cost_usd - before.total_cost_usd),
        "latency_seconds": max(
            0.0,
            after.total_latency_seconds - before.total_latency_seconds,
        ),
    }


def provider_route_for_oracles(
    *,
    mode: OracleMode,
    oracle_names: Sequence[str],
) -> str:
    """Return coarse provider route tag used in row/manifest metadata."""

    if mode == "fixture":
        return "fixture"
    if "llm_judge" in {str(name).strip() for name in oracle_names}:
        return "openrouter"
    return "local_models"


def fixture_oracles(names: Sequence[str] | None = None) -> list[FixtureOracle]:
    """Return deterministic local oracle objects by canonical name."""

    selected = list(names or ["nli", "semantic_similarity", "llm_judge"])
    oracles: list[FixtureOracle] = []
    for name in selected:
        query_weight = 0.25 if name == "llm_judge" else 0.0
        oracles.append(FixtureOracle(name, query_weight=query_weight))
    return oracles


def fixture_model_ids(names: Sequence[str] | None = None) -> list[str]:
    """Return fixture model IDs matching the selected oracle names."""

    selected = list(names or ["nli", "semantic_similarity", "llm_judge"])
    mapping = {
        "nli": "fixture-nli",
        "semantic_similarity": "fixture-semantic-similarity",
        "llm_judge": "fixture-llm-judge",
    }
    model_ids = ["fixture-fits-generator"]
    for name in selected:
        model_id = mapping.get(name, f"fixture-{name}")
        if model_id not in model_ids:
            model_ids.append(model_id)
    return model_ids


def build_experiment_aggregator(
    *,
    strategy: str,
    weights: Mapping[str, float] | None = None,
    min_score: float = 0.5,
    min_oracles: int = 2,
) -> Aggregator:
    """Build an aggregator from lightweight experiment config fields."""

    if strategy == "uniform":
        return UniformAggregator()
    resolved_weights = dict(
        weights
        or {
            "nli": 0.4,
            "semantic_similarity": 0.3,
            "llm_judge": 0.3,
        }
    )
    if strategy == "confidence_gated":
        return ConfidenceGatedAggregator(
            weights=resolved_weights,
            min_score=min_score,
            min_passing_oracles=min_oracles,
        )
    if strategy == "weighted":
        return WeightedAggregator(resolved_weights)
    msg = f"Unsupported aggregation strategy: {strategy}"
    raise ValueError(msg)


def estimate_cost_usd(
    settings: ExperimentRunSettings,
    *,
    work_units_per_record: int = 1,
) -> float:
    """Estimate run cost for the cost confirmation gate."""

    return (
        settings.estimated_cost_per_example_usd
        * settings.query_limit
        * len(settings.seeds)
        * work_units_per_record
    )


def enforce_cost_gate(
    settings: ExperimentRunSettings,
    *,
    estimated_cost_usd: float,
    confirm_cost: bool,
) -> None:
    """Block runs whose estimated cost exceeds the configured cap."""

    daily_cap = _openrouter_daily_cap()
    cap_candidates = [POLICY_CONFIRM_COST_CAP_USD]
    if settings.cost_cap_usd > 0:
        cap_candidates.append(settings.cost_cap_usd)
    if daily_cap is not None and daily_cap > 0:
        cap_candidates.append(daily_cap)
    effective_cap = min(cap_candidates)

    if effective_cap <= 0.0:
        return
    if estimated_cost_usd <= effective_cap:
        return
    if confirm_cost:
        return
    msg = (
        "Cost gate blocked run: estimated "
        f"${estimated_cost_usd:.4f} exceeds cap ${effective_cap:.4f}. "
        "Re-run with --confirm-cost after reviewing the estimate."
    )
    raise RuntimeError(msg)


def print_cost_estimate(
    settings: ExperimentRunSettings,
    *,
    estimated_cost_usd: float,
) -> None:
    """Print the cost estimate required before starting full runs."""

    daily_cap = _openrouter_daily_cap()
    cap_parts = [
        f"policy cap ${POLICY_CONFIRM_COST_CAP_USD:.4f}",
        f"configured cap ${settings.cost_cap_usd:.4f}",
    ]
    if daily_cap is not None:
        cap_parts.append(f"OPENROUTER_DAILY_USD_CAP ${daily_cap:.4f}")
    print(
        "Estimated experiment cost: "
        f"${estimated_cost_usd:.4f} for {settings.experiment_id} "
        f"({settings.mode}; {settings.query_limit} queries x "
        f"{len(settings.seeds)} seeds; {', '.join(cap_parts)})."
    )


def ensure_full_run_allowed(
    settings: ExperimentRunSettings,
    *,
    paths: ArtifactPaths,
    confirmed_smoke: bool,
) -> None:
    """Require an explicit smoke handoff before full experiment runs."""

    if settings.mode != "full" or not settings.require_smoke_before_full:
        return
    smoke_manifest = paths.manifest_json.with_name(
        paths.manifest_json.name.replace("_full_", "_smoke_")
    )
    if confirmed_smoke or smoke_manifest.exists():
        return
    msg = (
        "Full run blocked until the smoke manifest exists. "
        "Run smoke first or pass --confirmed-smoke."
    )
    raise RuntimeError(msg)


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> int:
    """Write JSONL rows and return the number written."""

    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(dict(row), sort_keys=True))
            file.write("\n")
            count += 1
    return count


def write_summary_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Write a compact CSV summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def write_manifest(
    *,
    settings: ExperimentRunSettings,
    paths: ArtifactPaths,
    status: str,
    row_count: int,
    failure_count: int,
    estimated_cost_usd: float,
    rows: Sequence[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the standard Phase 8 run manifest."""

    rows = [] if rows is None else list(rows)
    _write_duckdb_import_sql(paths)
    manifest = {
        "run_id": _run_id(settings=settings, paths=paths),
        "experiment_id": settings.experiment_id,
        "title": settings.title,
        "mode": settings.mode,
        "status": status,
        "dataset_path": str(settings.dataset_path),
        "dataset_checksum": _sha256_file(settings.dataset_path),
        "split": settings.split,
        "query_limit": settings.query_limit,
        "seeds": settings.seeds,
        "git_commit": _git_commit_hash(),
        "sdk_versions": _sdk_versions(),
        "model_ids": _collect_string_values(rows, "model_ids"),
        "provider_routing": _collect_string_values(rows, "provider_route"),
        "latency_seconds": round(_sum_numeric(rows, "latency_seconds"), 6),
        "token_counts": {
            "prompt_tokens": int(_sum_numeric(rows, "prompt_tokens")),
            "completion_tokens": int(_sum_numeric(rows, "completion_tokens")),
            "total_tokens": int(_sum_numeric(rows, "total_tokens")),
        },
        "estimated_cost_usd": round(estimated_cost_usd, 6),
        "cost_cap_usd": settings.cost_cap_usd,
        "openrouter_daily_usd_cap": _openrouter_daily_cap(),
        "row_count": row_count,
        "failure_count": failure_count,
        "raw_jsonl": str(paths.raw_jsonl),
        "summary_csv": str(paths.summary_csv),
        "duckdb_sql": str(paths.duckdb_sql),
        "failures_jsonl": str(paths.failures_jsonl),
        "config_snapshot_yaml": str(paths.config_snapshot_yaml),
        "written_at": datetime.now(UTC).isoformat(),
        "metadata": dict(metadata or {}),
    }
    paths.manifest_json.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def accuracy_summaries(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_key: str,
    experiment_id: str,
) -> list[dict[str, Any]]:
    """Summarize correctness by one grouping key plus an all-row aggregate."""

    summaries: list[dict[str, Any]] = []
    groups = sorted({str(row[group_key]) for row in rows if group_key in row})
    for group in groups:
        group_rows = [row for row in rows if str(row.get(group_key)) == group]
        summaries.append(_accuracy_row(group_key, group, group_rows, experiment_id))
    summaries.append(_accuracy_row(group_key, "all", rows, experiment_id))
    return summaries


def timed_seconds() -> float:
    """Return a monotonic timestamp for simple elapsed-time measurements."""

    return time.perf_counter()


def elapsed_since(started: float) -> float:
    """Return elapsed seconds from a monotonic timestamp."""

    return max(0.0, time.perf_counter() - started)


def _accuracy_row(
    group_key: str,
    group: str,
    rows: Sequence[Mapping[str, Any]],
    experiment_id: str,
) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for row in rows if row.get("correct") is True)
    return {
        "experiment_id": experiment_id,
        group_key: group,
        "examples": total,
        "correct": correct,
        "accuracy": round(correct / total, 6) if total else 0.0,
    }


def _mapping(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        msg = "Expected config section to be a mapping."
        raise ValueError(msg)
    return value


def _noise_text(record: Mapping[str, Any]) -> str | None:
    injection = record.get("injection")
    if not isinstance(injection, Mapping):
        return None
    for key in ("inserted_passage", "noise_text", "replacement_passage"):
        value = injection.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    removed = injection.get("removed_doc_id")
    if isinstance(removed, str) and removed.strip():
        return f"Retrieved context omitted supporting document {removed}."
    return None


def _support_text(record: Mapping[str, Any], gt_answer: str) -> str:
    for key in (
        "supporting_passage",
        "support_passage",
        "evidence_passage",
        "retrieved_passage",
    ):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"{gt_answer}. This passage directly supports the answer."


def _distractor_text(record: Mapping[str, Any]) -> str:
    for key in ("distractor_passage", "negative_passage", "noise_passage"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _noise_text(record) or (
        "Unrelated context about dataset curation and archive formats."
    )


def _fault_answer(record: Mapping[str, Any], gt_answer: str) -> str | None:
    for key in ("fault_answer", "hallucinated_answer", "answer_wrong", "fake_answer"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if str(record.get("fault_stage", "")).strip() == "retrieval":
        return f"The context does not support {gt_answer}."
    return None


def _openrouter_daily_cap() -> float | None:
    value = os.getenv("OPENROUTER_DAILY_USD_CAP")
    if not value:
        return None
    return float(value)


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _sdk_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
    }
    for package in ("mutoracle-rag", "openai", "pydantic", "typer", "pyyaml"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def _collect_string_values(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        raw = row.get(key)
        candidates = raw if isinstance(raw, list) else [raw]
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            value = candidate.strip()
            if value in seen:
                continue
            seen.add(value)
            values.append(value)
    return values


def _sum_numeric(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    total = 0.0
    for row in rows:
        value = row.get(key, 0.0)
        if isinstance(value, int | float):
            total += float(value)
    return total


def _run_id(*, settings: ExperimentRunSettings, paths: ArtifactPaths) -> str:
    payload = {
        "config": _sha256_file(settings.config_path),
        "dataset": _sha256_file(settings.dataset_path),
        "experiment_id": settings.experiment_id,
        "mode": settings.mode,
        "raw": _sha256_file(paths.raw_jsonl),
        "seeds": settings.seeds,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _write_duckdb_import_sql(paths: ArtifactPaths) -> None:
    paths.duckdb_sql.parent.mkdir(parents=True, exist_ok=True)
    paths.duckdb_sql.write_text(
        "\n".join(
            [
                "-- DuckDB import helper generated by Phase 8 scripts.",
                "CREATE OR REPLACE VIEW raw_results AS",
                f"SELECT * FROM read_json_auto('{paths.raw_jsonl.as_posix()}');",
                "CREATE OR REPLACE VIEW summary_results AS",
                f"SELECT * FROM read_csv_auto('{paths.summary_csv.as_posix()}');",
                "",
            ]
        ),
        encoding="utf-8",
    )
