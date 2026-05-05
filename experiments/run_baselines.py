"""Run Phase 7 fixture baselines or Phase 8 config-driven detection experiments."""

from __future__ import annotations

import argparse
import json
from importlib import resources
from pathlib import Path
from typing import Any

from mutoracle.baselines import (
    BaselineExample,
    BaselineResult,
    LexicalNLIBackend,
    MetaRAGBaseline,
    NLIClaimVerifier,
    OfficialRagasFaithfulnessScorer,
    RagasBaseline,
    run_baselines,
    run_id_for,
    write_baseline_outputs,
)
from mutoracle.cache import SQLiteCacheLedger
from mutoracle.config import load_config
from mutoracle.contracts import RAGRun
from mutoracle.experiments import (
    accuracy_summaries,
    artifact_paths,
    build_experiment_aggregator,
    elapsed_since,
    enforce_cost_gate,
    ensure_full_run_allowed,
    estimate_cost_usd,
    expected_detection_label,
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
from mutoracle.localizer import FaultLocalizer, fault_report_to_dict
from mutoracle.mutations.base import content_similarity
from mutoracle.oracles.base import context_text
from mutoracle.rag import FixtureRAGPipeline


def main() -> None:
    """CLI entry point for the baseline smoke/fixture runner."""

    args = _parse_args()
    if args.experiment_config is not None:
        _run_phase8_experiment(args)
        return

    config = load_config(args.config)
    pipeline = FixtureRAGPipeline(config=config, corpus_path=args.corpus)
    examples = [
        BaselineExample(run=pipeline.run(query))
        for query in _fixture_queries(limit=args.queries)
    ]

    baselines: list[Any] = []
    if args.baseline in {"metarag", "all"}:
        baselines.append(
            MetaRAGBaseline(
                verifier=NLIClaimVerifier(
                    backend=LexicalNLIBackend(),
                    model_id="fixture-lexical-nli",
                )
            )
        )
    if args.baseline in {"ragas", "all"}:
        baselines.append(
            RagasBaseline(scorer=OfficialRagasFaithfulnessScorer(config=config))
        )

    thresholds = {baseline.name: args.threshold for baseline in baselines}
    results = run_baselines(
        examples=examples,
        baselines=baselines,
        thresholds=thresholds,
    )
    manifest = write_baseline_outputs(
        results=results,
        output_path=args.output,
        thresholds=thresholds,
        metadata={
            "script": "experiments/run_baselines.py",
            "queries": args.queries,
            "source": "packaged_fixture_queries",
        },
    )
    print(f"Wrote {len(results)} baseline rows to {args.output}")
    print(f"Wrote manifest for {manifest.run_count} runs")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        choices=["metarag", "ragas", "all"],
        default="metarag",
        help="Baseline to run.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=2,
        help="Number of fixture queries to score.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Faithfulness threshold for hallucination labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/baselines_smoke.jsonl"),
        help="JSONL output path.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config path.",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="Optional fixture corpus JSON path.",
    )
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=None,
        help="Optional Phase 8 baseline experiment YAML config.",
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "full"],
        default="smoke",
        help="Phase 8 run mode when --experiment-config is used.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Shortcut for --mode smoke when --experiment-config is used.",
    )
    parser.add_argument(
        "--confirm-cost",
        action="store_true",
        help="Allow Phase 8 runs above the configured cost cap.",
    )
    parser.add_argument(
        "--confirmed-smoke",
        action="store_true",
        help="Confirm Phase 8 smoke completion before a full run.",
    )
    args = parser.parse_args()
    if args.smoke:
        args.mode = "smoke"
    if args.queries < 1:
        parser.error("--queries must be at least 1")
    if args.threshold < 0.0 or args.threshold > 1.0:
        parser.error("--threshold must be in [0, 1]")
    return args


def _fixture_queries(*, limit: int) -> list[str]:
    raw_text = (
        resources.files("mutoracle.fixtures")
        .joinpath("queries.json")
        .read_text(encoding="utf-8")
    )
    queries = json.loads(raw_text)
    if not isinstance(queries, list) or not all(
        isinstance(query, str) for query in queries
    ):
        msg = "Packaged fixture queries must be a JSON list of strings."
        raise ValueError(msg)
    selected: list[str] = []
    while len(selected) < limit:
        selected.extend(queries)
    return selected[:limit]


def _run_phase8_experiment(args: argparse.Namespace) -> None:
    settings = resolve_run_settings(
        args.experiment_config,
        mode=args.mode,
        default_experiment_id="e1_detection",
    )
    raw_config = load_experiment_config(args.experiment_config)
    baseline_config = _section(raw_config, "baseline")
    baseline_names = [
        str(name).lower() for name in baseline_config.get("baselines", ["metarag"])
    ]
    oracle_mode = resolve_oracle_mode(baseline_config)
    runtime_config_path = resolve_runtime_config_path(
        raw_config,
        section=baseline_config,
    )
    runtime_config = None
    ledger: SQLiteCacheLedger | None = None
    if oracle_mode == "real":
        runtime_config = load_runtime_config(runtime_config_path)
        ledger = SQLiteCacheLedger(runtime_config.runtime.cache_path)

    paths = artifact_paths(settings)
    estimated_cost = estimate_cost_usd(
        settings,
        work_units_per_record=max(1, len(baseline_names)),
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
    thresholds = {
        name: float(baseline_config.get("threshold", 0.5)) for name in baseline_names
    }
    baselines = _phase8_baselines(
        baseline_names,
        evaluator=str(baseline_config.get("evaluator", "fixture")),
        oracle_mode=oracle_mode,
        runtime_config=runtime_config,
        ledger=ledger,
    )

    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed in settings.seeds:
        for record in records:
            run = rag_run_from_fits_record(record, seed=seed)
            expected_label = expected_detection_label(record)
            example = BaselineExample(
                run=run,
                reference=str(record["gt_answer"]),
                expected_label=expected_label,
                split=record["split"],
            )
            for baseline in baselines:
                started = timed_seconds()
                try:
                    result = baseline.run(
                        example.run,
                        threshold=thresholds[baseline.name],
                        reference=example.reference,
                    )
                    payload = result.model_dump(mode="json")
                    payload.update(
                        {
                            "experiment_id": settings.experiment_id,
                            "mode": settings.mode,
                            "seed": seed,
                            "qid": record["qid"],
                            "split": record["split"],
                            "fault_stage": record["fault_stage"],
                            "expected_label": expected_label,
                            "correct": result.predicted_label == expected_label,
                            "experiment_latency_seconds": round(
                                elapsed_since(started), 6
                            ),
                            "prompt_tokens": int(
                                _usage_metadata(result.metadata, "prompt_tokens")
                            ),
                            "completion_tokens": int(
                                _usage_metadata(result.metadata, "completion_tokens")
                            ),
                            "total_tokens": int(
                                _usage_metadata(result.metadata, "total_tokens")
                            ),
                            "provider_route": result.metadata.get(
                                "provider_route",
                                "fixture",
                            ),
                        }
                    )
                    rows.append(payload)
                except Exception as error:
                    failures.append(
                        {
                            "experiment_id": settings.experiment_id,
                            "mode": settings.mode,
                            "seed": seed,
                            "qid": record.get("qid"),
                            "baseline_name": baseline.name,
                            "reason": str(error),
                            "error_type": type(error).__name__,
                        }
                    )

    row_count = write_jsonl(rows, paths.raw_jsonl)
    failure_count = write_jsonl(failures, paths.failures_jsonl)
    summaries = accuracy_summaries(
        rows,
        group_key="baseline_name",
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
            "script": "experiments/run_baselines.py",
            "baseline_names": baseline_names,
            "thresholds": thresholds,
        },
    )
    print(f"Wrote {row_count} baseline rows to {paths.raw_jsonl}")
    print(f"Wrote manifest to {paths.manifest_json}")
    print(f"Recorded seeds: {manifest['seeds']}")


def _phase8_baselines(
    names: list[str],
    *,
    evaluator: str,
    oracle_mode: str,
    runtime_config: Any,
    ledger: SQLiteCacheLedger | None,
) -> list[Any]:
    baselines: list[Any] = []
    config_for_real = runtime_config
    for name in names:
        if name == "metarag":
            if evaluator != "fixture" and config_for_real is None:
                config_for_real = load_runtime_config(None)
            baselines.append(
                MetaRAGBaseline(
                    verifier=NLIClaimVerifier(
                        backend=(
                            LexicalNLIBackend() if evaluator == "fixture" else None
                        ),
                        model_id=(
                            "fixture-lexical-nli"
                            if evaluator == "fixture"
                            else str(config_for_real.oracles.nli_model)
                        ),
                    )
                )
            )
        elif name == "ragas":
            if evaluator == "fixture":
                baselines.append(RagasBaseline(scorer=FixtureRagasScorer()))
            else:
                if config_for_real is None:
                    config_for_real = load_runtime_config(None)
                baselines.append(
                    RagasBaseline(
                        scorer=OfficialRagasFaithfulnessScorer(config=config_for_real)
                    )
                )
        elif name in {"mutoracle", "mutoracle_weighted"}:
            baselines.append(
                MutOracleDetectionBaseline(
                    name=name,
                    oracle_mode=oracle_mode,
                    runtime_config=config_for_real,
                    ledger=ledger,
                )
            )
        else:
            msg = f"Unsupported baseline in experiment config: {name}"
            raise ValueError(msg)
    return baselines


class MutOracleDetectionBaseline:
    """Response-level MutOracle variant for E1 detection comparisons."""

    def __init__(
        self,
        *,
        name: str = "mutoracle_weighted",
        oracle_mode: str = "fixture",
        runtime_config: Any = None,
        ledger: SQLiteCacheLedger | None = None,
    ) -> None:
        self.name = name
        self._oracle_mode = oracle_mode
        self._runtime_config = runtime_config
        self._ledger = ledger

    def run(
        self,
        run: RAGRun,
        *,
        threshold: float = 0.5,
        reference: str | None = None,
    ) -> BaselineResult:
        del reference
        started = timed_seconds()
        usage_before = (
            self._ledger.usage_summary() if self._ledger is not None else None
        )
        oracle_names = ["nli", "semantic_similarity", "llm_judge"]
        configured_oracles = (
            fixture_oracles(oracle_names)
            if self._oracle_mode == "fixture"
            else real_oracles(
                oracle_names,
                config=self._runtime_config,
                ledger=self._ledger,
            )
        )
        localizer = FaultLocalizer(
            pipeline=FixedRunPipeline(run),
            oracles=configured_oracles,
            aggregator=build_experiment_aggregator(
                strategy="weighted",
                weights={
                    "nli": 0.4,
                    "semantic_similarity": 0.3,
                    "llm_judge": 0.3,
                },
            ),
            delta_threshold=0.05,
            seed=int(run.metadata.get("generation", {}).get("seed", 2026)),
        )
        report = localizer.diagnose(run.query)
        usage = (
            usage_delta(usage_before, self._ledger.usage_summary())
            if self._ledger is not None and usage_before is not None
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
        predicted_label = (
            "faithful" if report.stage == "no_fault_detected" else "hallucinated"
        )
        score = (
            1.0 if predicted_label == "faithful" else max(0.0, 1.0 - report.confidence)
        )
        generation_model = str(
            run.metadata.get("generation", {}).get("model", "fixture")
        )
        model_ids = (
            [
                generation_model,
                "fixture-nli",
                "fixture-semantic-similarity",
                "fixture-llm-judge",
            ]
            if self._runtime_config is None
            else real_model_ids(
                oracle_names,
                config=self._runtime_config,
                generation_model=generation_model,
            )
        )
        provider_route = provider_route_for_oracles(
            mode="fixture" if self._runtime_config is None else "real",
            oracle_names=oracle_names,
        )
        return BaselineResult(
            run_id=run_id_for(run),
            baseline_name=self.name,
            query=run.query,
            score=score,
            threshold=threshold,
            predicted_label=predicted_label,
            latency_seconds=elapsed_since(started),
            cost_usd=max(0.0, float(usage["cost_usd"])),
            model_ids=model_ids,
            scores={
                "faithfulness": score,
                "fault_confidence": report.confidence,
            },
            metadata={
                "stage": report.stage,
                "stage_deltas": report.stage_deltas,
                "operator_deltas": fault_report_to_dict(report)["deltas"],
                "provider_route": provider_route,
                "cost_scope": (
                    "oracle_usage"
                    if self._runtime_config is not None
                    else "fixture_zero_cost"
                ),
                "usage": {
                    "prompt_tokens": int(usage["prompt_tokens"]),
                    "completion_tokens": int(usage["completion_tokens"]),
                    "total_tokens": int(usage["total_tokens"]),
                    "provider_latency_seconds": float(usage["latency_seconds"]),
                },
            },
        )


class FixedRunPipeline:
    """RAGPipeline adapter that reruns a fixed result with the requested query."""

    def __init__(self, run: RAGRun) -> None:
        self._run = run

    def run(self, query: str) -> RAGRun:
        return RAGRun(
            query=query,
            passages=self._run.passages,
            answer=self._run.answer,
            metadata=self._run.metadata,
        )


class FixtureRagasScorer:
    """Credential-free RAGAS-shaped scorer for smoke experiment records."""

    model_id = "fixture-ragas-judge"

    def score(self, run: RAGRun, *, reference: str | None = None) -> float:
        return self.score_metrics(run, reference=reference)["faithfulness"]

    def score_metrics(
        self,
        run: RAGRun,
        *,
        reference: str | None = None,
    ) -> dict[str, float]:
        context = context_text(run)
        reference_text = reference or run.answer
        faithfulness = content_similarity(context, run.answer)
        return {
            "faithfulness": faithfulness,
            "answer_relevancy": content_similarity(run.query, run.answer),
            "context_precision": content_similarity(context, reference_text),
            "context_recall": content_similarity(reference_text, context),
        }


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        msg = f"{name} config section must be a mapping."
        raise ValueError(msg)
    return value


def _usage_metadata(metadata: dict[str, Any], key: str) -> int | float:
    usage = metadata.get("usage", {})
    if not isinstance(usage, dict):
        return 0
    value = usage.get(key, 0)
    if isinstance(value, int | float):
        return value
    return 0


if __name__ == "__main__":
    main()
