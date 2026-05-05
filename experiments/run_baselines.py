"""Run Phase 7 response-level baselines on shared fixture RAG outputs."""

from __future__ import annotations

import argparse
import json
from importlib import resources
from pathlib import Path
from typing import Any

from mutoracle.baselines import (
    BaselineExample,
    LexicalNLIBackend,
    MetaRAGBaseline,
    NLIClaimVerifier,
    OfficialRagasFaithfulnessScorer,
    RagasBaseline,
    run_baselines,
    write_baseline_outputs,
)
from mutoracle.config import load_config
from mutoracle.rag import FixtureRAGPipeline


def main() -> None:
    """CLI entry point for the baseline smoke/fixture runner."""

    args = _parse_args()
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
    args = parser.parse_args()
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


if __name__ == "__main__":
    main()
