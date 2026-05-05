"""Run Phase 8 cost and latency reporting experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import fmean
from typing import Any

from mutoracle.cache import SQLiteCacheLedger
from mutoracle.experiments import (
    FITSRecordPipeline,
    artifact_paths,
    build_experiment_aggregator,
    elapsed_since,
    enforce_cost_gate,
    ensure_full_run_allowed,
    estimate_cost_usd,
    fixture_model_ids,
    fixture_oracles,
    load_experiment_config,
    load_runtime_config,
    print_cost_estimate,
    provider_route_for_oracles,
    rag_run_from_fits_record,
    real_model_ids,
    real_oracles,
    resolve_oracle_mode,
    resolve_run_settings,
    resolve_runtime_config_path,
    selected_fits_records,
    snapshot_config,
    timed_seconds,
    usage_delta,
    write_jsonl,
    write_manifest,
    write_summary_csv,
)
from mutoracle.localizer import FaultLocalizer


def main() -> None:
    args = _parse_args()
    settings = resolve_run_settings(
        args.config,
        mode=args.mode,
        default_experiment_id="e5_latency",
    )
    raw_config = load_experiment_config(args.config)
    latency_config = _section(raw_config, "latency")
    paths = artifact_paths(settings)

    estimated_cost = estimate_cost_usd(
        settings,
        work_units_per_record=int(latency_config.get("work_units_per_record", 2)),
    )
    print_cost_estimate(settings, estimated_cost_usd=estimated_cost)
    enforce_cost_gate(
        settings,
        estimated_cost_usd=estimated_cost,
        confirm_cost=args.confirm_cost,
    )
    ensure_full_run_allowed(
        settings,
        paths=paths,
        confirmed_smoke=args.confirmed_smoke,
    )
    snapshot_config(settings, paths)

    records = selected_fits_records(settings)
    oracle_names = [
        str(name)
        for name in latency_config.get(
            "oracles",
            ["nli", "semantic_similarity", "llm_judge"],
        )
    ]
    oracle_mode = resolve_oracle_mode(latency_config)
    runtime_config = None
    ledger: SQLiteCacheLedger | None = None
    if oracle_mode == "real":
        runtime_config = load_runtime_config(
            resolve_runtime_config_path(raw_config, section=latency_config)
        )
        ledger = SQLiteCacheLedger(runtime_config.runtime.cache_path)

    configured_oracles = (
        fixture_oracles(oracle_names)
        if oracle_mode == "fixture"
        else real_oracles(oracle_names, config=runtime_config, ledger=ledger)
    )
    localizer_provider_route = provider_route_for_oracles(
        mode=oracle_mode,
        oracle_names=oracle_names,
    )
    localizer_model_ids = (
        fixture_model_ids(oracle_names)
        if runtime_config is None
        else real_model_ids(oracle_names, config=runtime_config)
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in settings.seeds:
        for record in records:
            try:
                rag_started = timed_seconds()
                run = rag_run_from_fits_record(record, seed=seed)
                rag_latency = elapsed_since(rag_started)
                generation = run.metadata.get("generation", {})
                usage = (
                    generation.get("usage", {}) if isinstance(generation, dict) else {}
                )
                rows.append(
                    {
                        "experiment_id": settings.experiment_id,
                        "mode": settings.mode,
                        "workflow": "rag_fixture",
                        "seed": seed,
                        "qid": record["qid"],
                        "latency_seconds": round(rag_latency, 6),
                        "cost_usd": round(
                            float(generation.get("estimated_cost_usd", 0.0)),
                            8,
                        ),
                        "overhead_vs_rag": 1.0,
                        "model_ids": [
                            str(
                                generation.get(
                                    "model",
                                    "fixture-fits-generator",
                                )
                            )
                        ],
                        "provider_route": str(
                            generation.get("provider_route", "fixture")
                        ),
                        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                        "completion_tokens": int(usage.get("completion_tokens", 0)),
                        "total_tokens": int(usage.get("total_tokens", 0)),
                    }
                )

                usage_before = ledger.usage_summary() if ledger is not None else None
                localizer_started = timed_seconds()
                localizer = FaultLocalizer(
                    pipeline=FITSRecordPipeline(record, seed=seed),
                    oracles=configured_oracles,
                    aggregator=build_experiment_aggregator(
                        strategy=str(latency_config.get("aggregation", "weighted")),
                        weights=latency_config.get(
                            "weights",
                            {
                                "nli": 0.4,
                                "semantic_similarity": 0.3,
                                "llm_judge": 0.3,
                            },
                        ),
                    ),
                    delta_threshold=float(latency_config.get("delta_threshold", 0.05)),
                    seed=seed,
                )
                localizer.diagnose(str(record["query"]))
                localizer_usage = (
                    usage_delta(usage_before, ledger.usage_summary())
                    if ledger is not None and usage_before is not None
                    else {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "cost_usd": 0.0,
                        "requests": 0,
                        "cache_hits": 0,
                        "live_requests": 0,
                        "latency_seconds": 0.0,
                    }
                )
                localizer_latency = elapsed_since(localizer_started)
                denominator = max(rag_latency, 1e-9)
                rows.append(
                    {
                        "experiment_id": settings.experiment_id,
                        "mode": settings.mode,
                        "workflow": "mutoracle_localizer",
                        "seed": seed,
                        "qid": record["qid"],
                        "latency_seconds": round(localizer_latency, 6),
                        "cost_usd": round(float(localizer_usage["cost_usd"]), 8),
                        "overhead_vs_rag": round(localizer_latency / denominator, 6),
                        "model_ids": localizer_model_ids,
                        "provider_route": localizer_provider_route,
                        "prompt_tokens": int(localizer_usage["prompt_tokens"]),
                        "completion_tokens": int(localizer_usage["completion_tokens"]),
                        "total_tokens": int(localizer_usage["total_tokens"]),
                        "provider_latency_seconds": round(
                            float(localizer_usage["latency_seconds"]),
                            6,
                        ),
                    }
                )
            except Exception as error:
                failures.append(
                    {
                        "experiment_id": settings.experiment_id,
                        "mode": settings.mode,
                        "seed": seed,
                        "qid": record.get("qid"),
                        "reason": str(error),
                        "error_type": type(error).__name__,
                    }
                )

    row_count = write_jsonl(rows, paths.raw_jsonl)
    failure_count = write_jsonl(failures, paths.failures_jsonl)
    write_summary_csv(_latency_summary(rows, settings.experiment_id), paths.summary_csv)
    manifest = write_manifest(
        settings=settings,
        paths=paths,
        status="complete" if failure_count == 0 else "complete_with_failures",
        row_count=row_count,
        failure_count=failure_count,
        estimated_cost_usd=estimated_cost,
        rows=rows,
        metadata={"script": "experiments/run_latency.py"},
    )
    print(f"Wrote {row_count} latency rows to {paths.raw_jsonl}")
    print(f"Wrote manifest to {paths.manifest_json}")
    print(f"Recorded seeds: {manifest['seeds']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/e5_latency.yaml"),
    )
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument(
        "--smoke", action="store_true", help="Shortcut for --mode smoke."
    )
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument("--confirmed-smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.mode = "smoke"
    return args


def _latency_summary(
    rows: list[dict[str, Any]],
    experiment_id: str,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for workflow in sorted({row["workflow"] for row in rows}):
        selected = [row for row in rows if row["workflow"] == workflow]
        summaries.append(
            {
                "experiment_id": experiment_id,
                "workflow": workflow,
                "examples": len(selected),
                "mean_latency_seconds": round(
                    fmean(float(row["latency_seconds"]) for row in selected),
                    6,
                ),
                "mean_cost_usd": round(
                    fmean(float(row["cost_usd"]) for row in selected),
                    6,
                ),
                "mean_overhead_vs_rag": round(
                    fmean(float(row["overhead_vs_rag"]) for row in selected),
                    6,
                ),
            }
        )
    return summaries


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        msg = f"{name} config section must be a mapping."
        raise ValueError(msg)
    return value


if __name__ == "__main__":
    main()
