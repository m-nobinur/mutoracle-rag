"""Shared schemas for response-level baseline comparisons."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from mutoracle.contracts import RAGRun

BaselineLabel = Literal["faithful", "hallucinated"]


class BaselineResult(BaseModel):
    """One response-level baseline result for a shared RAG output."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    baseline_name: str
    query: str
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    predicted_label: BaselineLabel
    latency_seconds: float = Field(ge=0.0)
    cost_usd: float = Field(ge=0.0)
    model_ids: list[str]
    scores: dict[str, float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaselineExample(BaseModel):
    """Input object used by baseline runners and calibration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    run: RAGRun
    reference: str | None = None
    expected_label: BaselineLabel | None = None
    split: Literal["validation", "test"] | None = None


class BaselineManifest(BaseModel):
    """Batch-level metadata for a baseline result file."""

    model_config = ConfigDict(extra="forbid")

    baseline_names: list[str]
    run_count: int
    thresholds: dict[str, float]
    result_schema: str = "mutoracle.baseline_result.v1"
    result_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def run_id_for(run: RAGRun) -> str:
    """Return a stable ID for a shared RAG output."""

    payload = {
        "answer": run.answer,
        "passages": run.passages,
        "query": run.query,
    }
    encoded = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def classify_faithfulness(*, score: float, threshold: float) -> BaselineLabel:
    """Convert a faithfulness score into the shared hallucination label."""

    return "hallucinated" if score < threshold else "faithful"


def run_metadata_value(run: RAGRun, path: tuple[str, ...], default: float) -> float:
    """Read a numeric metadata value from a nested RAGRun metadata path."""

    current: Any = run.metadata
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    if isinstance(current, int | float):
        return float(current)
    return default


def run_metadata_model_ids(
    run: RAGRun,
    path: tuple[str, ...] = ("generation", "model"),
) -> list[str]:
    """Read one or many model IDs from nested RAGRun metadata."""

    current: Any = run.metadata
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return []
        current = current[key]

    if isinstance(current, str):
        stripped = current.strip()
        return [stripped] if stripped else []
    if isinstance(current, list):
        return [
            item.strip() for item in current if isinstance(item, str) and item.strip()
        ]
    return []


def merge_model_ids(*groups: Sequence[str]) -> list[str]:
    """Merge model IDs while preserving order and removing duplicates."""

    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for model_id in group:
            normalized = model_id.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            merged.append(normalized)
    return merged
