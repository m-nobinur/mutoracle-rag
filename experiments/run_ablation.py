"""Run Phase 8 oracle and aggregation ablation experiments."""

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
from mutoracle.mutations import mutation_registry


def main() -> None:
    args = _parse_args()
    settings = resolve_run_settings(
        args.config,
        mode=args.mode,
        default_experiment_id="e3_ablation",
    )
    settings = override_run_settings(
        settings,
        query_limit=args.query_limit,
        seeds=args.seeds,
    )
    raw_config = load_experiment_config(args.config)
    ablation_config = _section(raw_config, "ablation")
    variants = _variants(ablation_config)
    paths = artifact_paths(settings)

    estimated_cost = estimate_cost_usd(
        settings,
        work_units_per_record=max(1, len(variants)),
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
    oracle_mode = resolve_oracle_mode(ablation_config)
    runtime_config = None
    ledger: SQLiteCacheLedger | None = None
    if oracle_mode == "real":
        runtime_config = load_runtime_config(
            resolve_runtime_config_path(raw_config, section=ablation_config)
        )
        ledger = SQLiteCacheLedger(runtime_config.runtime.cache_path)

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    record_count = len(records)
    total = len(variants) * len(settings.seeds) * record_count
    completed = 0
    run_started = timed_seconds()
    print(
        f"Running {settings.experiment_id} mode={settings.mode} "
        f"variants={len(variants)} records={record_count} "
        f"seeds={settings.seeds} total_rows={total}",
        flush=True,
    )
    for variant in variants:
        configured_oracles = (
            fixture_oracles(variant["oracles"])
            if oracle_mode == "fixture"
            else real_oracles(variant["oracles"], config=runtime_config, ledger=ledger)
        )
        provider_route = provider_route_for_oracles(
            mode=oracle_mode,
            oracle_names=variant["oracles"],
        )
        model_ids = (
            fixture_model_ids(variant["oracles"])
            if runtime_config is None
            else real_model_ids(variant["oracles"], config=runtime_config)
        )
        for seed in settings.seeds:
            seed_started = timed_seconds()
            seed_rows_before = len(rows)
            seed_failures_before = len(failures)
            print(
                "Running ablation "
                f"variant={variant['name']} seed={seed} records={record_count}"
            )
            for record in records:
                started = timed_seconds()
                try:
                    usage_before = (
                        ledger.usage_summary() if ledger is not None else None
                    )
                    pipeline = FITSRecordPipeline(record, seed=seed)
                    localizer = FaultLocalizer(
                        pipeline=pipeline,
                        oracles=configured_oracles,
                        aggregator=build_experiment_aggregator(
                            strategy=variant["aggregation"],
                            weights=variant.get("weights"),
                            min_score=float(variant.get("min_score", 0.5)),
                            min_oracles=int(variant.get("min_oracles", 2)),
                        ),
                        delta_threshold=float(variant.get("delta_threshold", 0.05)),
                        operators=_operator_subset(variant.get("operators")),
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
                    payload = fault_report_to_dict(report)
                    rows.append(
                        {
                            "experiment_id": settings.experiment_id,
                            "mode": settings.mode,
                            "ablation_name": variant["name"],
                            "aggregation": variant["aggregation"],
                            "oracles": variant["oracles"],
                            "seed": seed,
                            "qid": record["qid"],
                            "split": record["split"],
                            "fault_stage": record["fault_stage"],
                            "expected_stage": expected_stage,
                            "predicted_stage": report.stage,
                            "correct": report.stage == expected_stage,
                            "confidence": round(report.confidence, 6),
                            "operator_deltas": payload["deltas"],
                            "stage_deltas": payload["stage_deltas"],
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
                            "ablation_name": variant["name"],
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

            print(
                "Completed ablation "
                f"variant={variant['name']} seed={seed} "
                f"rows={len(rows) - seed_rows_before} "
                f"failures={len(failures) - seed_failures_before} "
                f"elapsed_seconds={elapsed_since(seed_started):.3f}"
            )

    row_count = write_jsonl(rows, paths.raw_jsonl)
    failure_count = write_jsonl(failures, paths.failures_jsonl)
    summaries = accuracy_summaries(
        rows,
        group_key="ablation_name",
        experiment_id=settings.experiment_id,
    ) + accuracy_summaries(
        rows,
        group_key="seed",
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
            "script": "experiments/run_ablation.py",
            "variant_names": [variant["name"] for variant in variants],
        },
    )
    print(f"Wrote {row_count} ablation rows to {paths.raw_jsonl}")
    print(f"Wrote manifest to {paths.manifest_json}")
    print(f"Recorded seeds: {manifest['seeds']}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/e3_ablation.yaml"),
    )
    parser.add_argument("--mode", choices=["smoke", "dev", "full"], default="smoke")
    parser.add_argument(
        "--smoke", action="store_true", help="Shortcut for --mode smoke."
    )
    parser.add_argument("--confirm-cost", action="store_true")
    parser.add_argument("--confirmed-smoke", action="store_true")
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


def _variants(config: dict[str, Any]) -> list[dict[str, Any]]:
    raw_variants = config.get("variants", [])
    if not isinstance(raw_variants, list) or not raw_variants:
        msg = "ablation.variants must be a non-empty list."
        raise ValueError(msg)
    variants: list[dict[str, Any]] = []
    for raw in raw_variants:
        if not isinstance(raw, dict):
            msg = "Each ablation variant must be a mapping."
            raise ValueError(msg)
        name = str(raw.get("name", "")).strip()
        if not name:
            msg = "Each ablation variant needs a name."
            raise ValueError(msg)
        oracles = [str(item) for item in raw.get("oracles", ["nli"])]
        variants.append(
            {
                "name": name,
                "oracles": oracles,
                "aggregation": str(raw.get("aggregation", "uniform")),
                "weights": raw.get("weights"),
                "operators": raw.get("operators"),
                "delta_threshold": raw.get("delta_threshold", 0.05),
                "min_score": raw.get("min_score", 0.5),
                "min_oracles": raw.get("min_oracles", 2),
            }
        )
    return variants


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        msg = f"{name} config section must be a mapping."
        raise ValueError(msg)
    return value


def _operator_subset(operator_ids: object) -> dict[str, Any] | None:
    if operator_ids is None:
        return None
    if not isinstance(operator_ids, list):
        msg = "ablation variant operators must be a list when provided."
        raise ValueError(msg)
    requested = {str(operator_id).upper() for operator_id in operator_ids}
    registry = mutation_registry()
    unknown = sorted(requested - set(registry))
    if unknown:
        msg = f"Unknown mutation operators in ablation variant: {unknown}"
        raise ValueError(msg)
    return {
        operator_id: operator
        for operator_id, operator in registry.items()
        if operator_id in requested
    }


if __name__ == "__main__":
    main()
