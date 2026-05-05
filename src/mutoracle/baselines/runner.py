"""Shared runner for comparable baseline result files."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from mutoracle.baselines.schema import (
    BaselineExample,
    BaselineManifest,
    BaselineResult,
)
from mutoracle.contracts import RAGRun


class Baseline(Protocol):
    """Protocol implemented by all response-level baselines."""

    name: str

    def run(
        self,
        run: RAGRun,
        *,
        threshold: float = 0.5,
        reference: str | None = None,
    ) -> BaselineResult:
        """Score one shared RAG output."""


def run_baselines(
    *,
    examples: Sequence[BaselineExample],
    baselines: Sequence[Baseline],
    thresholds: Mapping[str, float] | None = None,
) -> list[BaselineResult]:
    """Run each baseline on each shared RAG output."""

    thresholds = {} if thresholds is None else thresholds
    results: list[BaselineResult] = []
    for example in examples:
        for baseline in baselines:
            results.append(
                baseline.run(
                    example.run,
                    threshold=float(thresholds.get(baseline.name, 0.5)),
                    reference=example.reference,
                )
            )
    return results


def write_baseline_outputs(
    *,
    results: Sequence[BaselineResult],
    output_path: Path,
    thresholds: Mapping[str, float],
    metadata: Mapping[str, object] | None = None,
) -> BaselineManifest:
    """Write JSONL results and a sidecar manifest."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result.model_dump(mode="json"), sort_keys=True))
            file.write("\n")

    manifest = BaselineManifest(
        baseline_names=sorted({result.baseline_name for result in results}),
        run_count=len({result.run_id for result in results}),
        thresholds=dict(thresholds),
        result_path=str(output_path),
        metadata=dict(metadata or {}),
    )
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
