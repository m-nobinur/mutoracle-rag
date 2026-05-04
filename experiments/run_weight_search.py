"""Deterministic grid search for Phase 5 aggregation calibration."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mutoracle.aggregation import WeightedAggregator
from mutoracle.config import load_config
from mutoracle.contracts import DiagnosisStage, Stage
from mutoracle.localizer import choose_stage

ORACLE_NAMES = ("nli", "semantic_similarity", "llm_judge")
OPERATOR_STAGES: dict[str, Stage] = {
    "CI": "retrieval",
    "CR": "retrieval",
    "CS": "retrieval",
    "QP": "prompt",
    "QN": "prompt",
    "FS": "generation",
    "FA": "generation",
}
THRESHOLD_CANDIDATES = (0.01, 0.03, 0.05, 0.08, 0.1)
DEFAULT_WEIGHTS = {"nli": 0.4, "semantic_similarity": 0.3, "llm_judge": 0.3}


@dataclass(frozen=True)
class CalibrationExample:
    """One synthetic labeled calibration example."""

    label: DiagnosisStage
    baseline: dict[str, float]
    mutated: dict[str, dict[str, float]]


CALIBRATION_EXAMPLES = (
    CalibrationExample(
        label="retrieval",
        baseline={"nli": 0.86, "semantic_similarity": 0.82, "llm_judge": 0.88},
        mutated={
            "CI": {"nli": 0.73, "semantic_similarity": 0.72, "llm_judge": 0.75},
            "CR": {"nli": 0.55, "semantic_similarity": 0.58, "llm_judge": 0.50},
            "CS": {"nli": 0.78, "semantic_similarity": 0.76, "llm_judge": 0.80},
            "QP": {"nli": 0.84, "semantic_similarity": 0.81, "llm_judge": 0.86},
            "QN": {"nli": 0.83, "semantic_similarity": 0.80, "llm_judge": 0.84},
            "FS": {"nli": 0.82, "semantic_similarity": 0.80, "llm_judge": 0.83},
            "FA": {"nli": 0.81, "semantic_similarity": 0.78, "llm_judge": 0.80},
        },
    ),
    CalibrationExample(
        label="prompt",
        baseline={"nli": 0.82, "semantic_similarity": 0.80, "llm_judge": 0.84},
        mutated={
            "CI": {"nli": 0.80, "semantic_similarity": 0.78, "llm_judge": 0.82},
            "CR": {"nli": 0.79, "semantic_similarity": 0.77, "llm_judge": 0.80},
            "CS": {"nli": 0.81, "semantic_similarity": 0.79, "llm_judge": 0.83},
            "QP": {"nli": 0.74, "semantic_similarity": 0.72, "llm_judge": 0.70},
            "QN": {"nli": 0.58, "semantic_similarity": 0.56, "llm_judge": 0.49},
            "FS": {"nli": 0.80, "semantic_similarity": 0.78, "llm_judge": 0.81},
            "FA": {"nli": 0.78, "semantic_similarity": 0.77, "llm_judge": 0.79},
        },
    ),
    CalibrationExample(
        label="generation",
        baseline={"nli": 0.88, "semantic_similarity": 0.84, "llm_judge": 0.90},
        mutated={
            "CI": {"nli": 0.85, "semantic_similarity": 0.81, "llm_judge": 0.87},
            "CR": {"nli": 0.84, "semantic_similarity": 0.80, "llm_judge": 0.86},
            "CS": {"nli": 0.87, "semantic_similarity": 0.83, "llm_judge": 0.89},
            "QP": {"nli": 0.86, "semantic_similarity": 0.82, "llm_judge": 0.88},
            "QN": {"nli": 0.84, "semantic_similarity": 0.81, "llm_judge": 0.85},
            "FS": {"nli": 0.75, "semantic_similarity": 0.73, "llm_judge": 0.78},
            "FA": {"nli": 0.43, "semantic_similarity": 0.52, "llm_judge": 0.36},
        },
    ),
    CalibrationExample(
        label="no_fault_detected",
        baseline={"nli": 0.81, "semantic_similarity": 0.80, "llm_judge": 0.82},
        mutated={
            "CI": {"nli": 0.80, "semantic_similarity": 0.79, "llm_judge": 0.81},
            "CR": {"nli": 0.79, "semantic_similarity": 0.78, "llm_judge": 0.80},
            "CS": {"nli": 0.81, "semantic_similarity": 0.80, "llm_judge": 0.82},
            "QP": {"nli": 0.80, "semantic_similarity": 0.79, "llm_judge": 0.81},
            "QN": {"nli": 0.79, "semantic_similarity": 0.78, "llm_judge": 0.80},
            "FS": {"nli": 0.80, "semantic_similarity": 0.79, "llm_judge": 0.81},
            "FA": {"nli": 0.78, "semantic_similarity": 0.77, "llm_judge": 0.79},
        },
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("experiments/configs/dev.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/configs/calibrated.yaml"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    result = run_weight_search(seed=args.seed)
    config = load_config(args.config)
    payload = config.model_dump(mode="json")
    payload["openrouter"]["api_key"] = None
    weights = dict(result["weights"])
    payload["aggregation"] = {
        **payload["aggregation"],
        "strategy": "weighted",
        "weights": weights,
        "delta_threshold": result["delta_threshold"],
    }
    payload["calibration"] = {**result, "weights": weights}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.dump(payload, Dumper=NoAliasDumper, sort_keys=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"Wrote calibrated config to {args.output}")


def run_weight_search(*, seed: int = 2026) -> dict[str, Any]:
    """Return the best deterministic grid-search calibration result."""

    best: dict[str, Any] | None = None
    for weights in candidate_weights():
        for threshold in THRESHOLD_CANDIDATES:
            predictions = [
                predict_example(example, weights=weights, threshold=threshold)
                for example in CALIBRATION_EXAMPLES
            ]
            correct = sum(
                prediction == example.label
                for prediction, example in zip(
                    predictions,
                    CALIBRATION_EXAMPLES,
                    strict=True,
                )
            )
            accuracy = correct / len(CALIBRATION_EXAMPLES)
            candidate = {
                "seed": seed,
                "weights": weights,
                "delta_threshold": threshold,
                "accuracy": accuracy,
                "examples": len(CALIBRATION_EXAMPLES),
                "predictions": predictions,
            }
            if best is None or _is_better(candidate, best):
                best = candidate
    if best is None:
        msg = "No calibration candidates were generated."
        raise RuntimeError(msg)
    return best


def candidate_weights() -> Iterable[dict[str, float]]:
    """Yield normalized 0.1-step weight triples in deterministic order."""

    for nli_units in range(11):
        for semantic_units in range(11 - nli_units):
            judge_units = 10 - nli_units - semantic_units
            yield {
                "nli": nli_units / 10,
                "semantic_similarity": semantic_units / 10,
                "llm_judge": judge_units / 10,
            }


def predict_example(
    example: CalibrationExample,
    *,
    weights: Mapping[str, float],
    threshold: float,
) -> DiagnosisStage:
    """Predict one calibration example with the final-plan decision rule."""

    aggregator = WeightedAggregator(dict(weights))
    baseline_score = aggregator.combine(example.baseline)
    deltas = {
        operator_id: baseline_score - aggregator.combine(scores)
        for operator_id, scores in example.mutated.items()
    }
    stage_deltas = {
        stage: max(
            deltas[operator_id]
            for operator_id, operator_stage in OPERATOR_STAGES.items()
            if operator_stage == stage
        )
        for stage in ("retrieval", "prompt", "generation")
    }
    return choose_stage(stage_deltas, delta_threshold=threshold)


def _is_better(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    return (
        candidate["accuracy"],
        -candidate["delta_threshold"],
        -_weight_distance(candidate["weights"]),
    ) > (
        current["accuracy"],
        -current["delta_threshold"],
        -_weight_distance(current["weights"]),
    )


def _weight_distance(weights: Mapping[str, float]) -> float:
    return sum(
        abs(float(weights[name]) - DEFAULT_WEIGHTS[name]) for name in ORACLE_NAMES
    )


class NoAliasDumper(yaml.SafeDumper):
    """YAML dumper that keeps calibration files explicit and diff-friendly."""

    def ignore_aliases(self, data: object) -> bool:
        return True


if __name__ == "__main__":
    main()
