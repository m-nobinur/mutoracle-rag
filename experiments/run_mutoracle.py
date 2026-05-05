"""Run Phase 8 MutOracle localization and separability experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mutoracle.cache import SQLiteCacheLedger
from mutoracle.experiments import (
    FITSRecordPipeline,
    accuracy_summaries,
    artifact_paths,
    build_experiment_aggregator,
    elapsed_since,
    enforce_cost_gate,
    ensure_full_run_allowed,
    estimate_cost_usd,
    expected_diagnosis_stage,
    fixture_model_ids,
    fixture_oracles,
    load_experiment_config,
    load_runtime_config,
    override_run_settings,
    print_cost_estimate,
    print_progress,
    provider_route_for_oracles,
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
from mutoracle.localizer import FaultLocalizer, fault_report_to_dict


def main() -> None:
    args = _parse_args()
    settings = resolve_run_settings(
        args.config,
        mode=args.mode,
        default_experiment_id="e2_localization",
    )
    settings = override_run_settings(
        settings,
        query_limit=args.query_limit,
        seeds=args.seeds,
    )
    raw_config = load_experiment_config(args.config)
    localizer_config = _section(raw_config, "localizer")
    paths = artifact_paths(settings)

    estimated_cost = estimate_cost_usd(
        settings,
        work_units_per_record=int(localizer_config.get("work_units_per_record", 8)),
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
        for name in localizer_config.get(
            "oracles",
            ["nli", "semantic_similarity", "llm_judge"],
        )
    ]
    oracle_mode = resolve_oracle_mode(localizer_config)
    runtime_config = None
    ledger: SQLiteCacheLedger | None = None
    if oracle_mode == "real":
        runtime_config = load_runtime_config(
            resolve_runtime_config_path(raw_config, section=localizer_config)
        )
        ledger = SQLiteCacheLedger(runtime_config.runtime.cache_path)

    configured_oracles = (
        fixture_oracles(oracle_names)
        if oracle_mode == "fixture"
        else real_oracles(oracle_names, config=runtime_config, ledger=ledger)
    )
    provider_route = provider_route_for_oracles(
        mode=oracle_mode,
        oracle_names=oracle_names,
    )
    model_ids = (
        fixture_model_ids(oracle_names)
        if runtime_config is None
        else real_model_ids(oracle_names, config=runtime_config)
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total = len(settings.seeds) * len(records)
    completed = 0
    run_started = timed_seconds()
    print(
        f"Running {settings.experiment_id} mode={settings.mode} "
        f"records={len(records)} seeds={settings.seeds} total_rows={total}",
        flush=True,
    )
    for seed in settings.seeds:
        for record in records:
            started = timed_seconds()
            try:
                usage_before = ledger.usage_summary() if ledger is not None else None
                pipeline = FITSRecordPipeline(record, seed=seed)
                localizer = FaultLocalizer(
                    pipeline=pipeline,
                    oracles=configured_oracles,
                    aggregator=build_experiment_aggregator(
                        strategy=str(localizer_config.get("aggregation", "weighted")),
                        weights=localizer_config.get("weights"),
                        min_score=float(localizer_config.get("min_score", 0.5)),
                        min_oracles=int(localizer_config.get("min_oracles", 2)),
                    ),
                    delta_threshold=float(
                        localizer_config.get("delta_threshold", 0.05)
                    ),
                    seed=seed,
                )
                report = localizer.diagnose(str(record["query"]))
                usage = (
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
                expected_stage = expected_diagnosis_stage(record)
                report_payload = fault_report_to_dict(report)
                rows.append(
                    {
                        "experiment_id": settings.experiment_id,
                        "mode": settings.mode,
                        "seed": seed,
                        "qid": record["qid"],
                        "split": record["split"],
                        "fault_stage": record["fault_stage"],
                        "expected_stage": expected_stage,
                        "predicted_stage": report.stage,
                        "correct": report.stage == expected_stage,
                        "confidence": round(report.confidence, 6),
                        "operator_deltas": report_payload["deltas"],
                        "stage_deltas": report_payload["stage_deltas"],
                        "model_ids": model_ids,
                        "provider_route": provider_route,
                        "prompt_tokens": int(usage["prompt_tokens"]),
                        "completion_tokens": int(usage["completion_tokens"]),
                        "total_tokens": int(usage["total_tokens"]),
                        "latency_seconds": round(elapsed_since(started), 6),
                        "provider_latency_seconds": round(
                            float(usage["latency_seconds"]),
                            6,
                        ),
                        "cost_usd": round(float(usage["cost_usd"]), 8),
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
            finally:
                completed += 1
                print_progress(
                    label=f"{settings.experiment_id} progress",
                    completed=completed,
                    total=total,
                    started_at=run_started,
                    every=args.progress_every,
                )

    row_count = write_jsonl(rows, paths.raw_jsonl)
    failure_count = write_jsonl(failures, paths.failures_jsonl)
    summaries = accuracy_summaries(
        rows,
        group_key="seed",
        experiment_id=settings.experiment_id,
    ) + accuracy_summaries(
        rows,
        group_key="expected_stage",
        experiment_id=settings.experiment_id,
    )
    write_summary_csv(summaries, paths.summary_csv)
    manifest = write_manifest(
        settings=settings,
        paths=paths,
        status="complete" if failure_count == 0 else "complete_with_failures",
        row_count=row_count,
        failure_count=failure_count,
        estimated_cost_usd=estimated_cost,
        rows=rows,
        metadata={
            "script": "experiments/run_mutoracle.py",
            "purpose": "E2 localization and E4 mutation-delta separability records",
        },
    )
    print(f"Wrote {row_count} rows to {paths.raw_jsonl}")
    print(f"Wrote manifest to {paths.manifest_json}")
    print(f"Recorded seeds: {manifest['seeds']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/e2_localization.yaml"),
        help="Phase 8 experiment YAML config.",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "dev", "full"],
        default="smoke",
        help="Run mode; full requires a smoke manifest or --confirmed-smoke.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shortcut for --mode smoke.",
    )
    parser.add_argument(
        "--confirm-cost",
        action="store_true",
        help="Allow runs whose cost estimate exceeds the configured cap.",
    )
    parser.add_argument(
        "--confirmed-smoke",
        action="store_true",
        help="Confirm that a smoke run has passed before a full run.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=None,
        help="Override the configured query limit for this run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override configured seeds for this run.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N diagnosed rows.",
    )
    args = parser.parse_args()
    if args.smoke:
        args.mode = "smoke"
    return args


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        msg = f"{name} config section must be a mapping."
        raise ValueError(msg)
    return value


if __name__ == "__main__":
    main()
