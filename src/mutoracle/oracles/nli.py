"""Natural-language inference oracle for context-answer faithfulness."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from mutoracle.cache import SQLiteCacheLedger
from mutoracle.config import MutOracleConfig
from mutoracle.contracts import RAGRun
from mutoracle.oracles.base import (
    CacheBackedOracle,
    OracleScore,
    clamp_score,
    context_text,
)


class NLIBackend(Protocol):
    """Minimal NLI backend used by the NLI oracle."""

    def probabilities(self, *, premise: str, hypothesis: str) -> dict[str, float]:
        """Return label probabilities keyed by lowercase NLI labels."""


class TransformersNLIBackend:
    """Lazy Hugging Face transformers NLI adapter."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline: Any | None = None

    def probabilities(self, *, premise: str, hypothesis: str) -> dict[str, float]:
        """Return label probabilities from a text-classification pipeline."""

        return self.probabilities_many([(premise, hypothesis)])[0]

    def probabilities_many(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[dict[str, float]]:
        """Return label probabilities for multiple premise/hypothesis pairs."""

        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as error:
                msg = (
                    "transformers is required for NLIOracle. Install the oracle "
                    "extras or inject an NLIBackend."
                )
                raise RuntimeError(msg) from error
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
            )

        output = self._pipeline(
            [
                {"text": premise, "text_pair": hypothesis}
                for premise, hypothesis in pairs
            ]
        )
        return [
            _probabilities_from_rows(rows)
            for rows in _split_pipeline_output(output, expected=len(pairs))
        ]


class NLIOracle(CacheBackedOracle):
    """Scores entailment probability for answer supported by context."""

    name = "nli"

    def __init__(
        self,
        *,
        config: MutOracleConfig | None = None,
        ledger: SQLiteCacheLedger | None = None,
        backend: NLIBackend | None = None,
        model_name: str | None = None,
    ) -> None:
        super().__init__(ledger=ledger)
        self.model_name = (
            model_name
            or (config.oracles.nli_model if config is not None else None)
            or "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        )
        self._backend = backend or TransformersNLIBackend(self.model_name)

    def _score_uncached(self, run: RAGRun, *, input_hash: str) -> OracleScore:
        return self._score_uncached_many([run], input_hashes=[input_hash])[0]

    def _score_uncached_many(
        self,
        runs: Sequence[RAGRun],
        *,
        input_hashes: Sequence[str],
    ) -> list[OracleScore]:
        prepared: list[tuple[RAGRun, str, str, str] | None] = []
        pairs: list[tuple[str, str]] = []
        for run, input_hash in zip(runs, input_hashes, strict=True):
            premise = context_text(run)
            hypothesis = run.answer.strip()
            if not premise or not hypothesis:
                prepared.append(None)
                continue
            prepared.append((run, input_hash, premise, hypothesis))
            pairs.append((premise, hypothesis))

        probabilities_by_pair = _backend_probabilities_many(self._backend, pairs)
        probability_index = 0
        results: list[OracleScore] = []
        for item, input_hash in zip(prepared, input_hashes, strict=True):
            if item is None:
                results.append(
                    OracleScore(
                        oracle_name=self.name,
                        value=0.0,
                        metadata={
                            "input_hash": input_hash,
                            "model": self.model_name,
                            "reason": "empty_premise_or_hypothesis",
                        },
                    )
                )
                continue
            probabilities = probabilities_by_pair[probability_index]
            probability_index += 1
            entailment_probability = _entailment_probability(probabilities)
            results.append(
                OracleScore(
                    oracle_name=self.name,
                    value=entailment_probability,
                    metadata={
                        "input_hash": item[1],
                        "model": self.model_name,
                        "label_probabilities": probabilities,
                    },
                )
            )
        return results


def _backend_probabilities_many(
    backend: NLIBackend,
    pairs: Sequence[tuple[str, str]],
) -> list[dict[str, float]]:
    if not pairs:
        return []
    probabilities_many = getattr(backend, "probabilities_many", None)
    if callable(probabilities_many):
        return list(probabilities_many(pairs))
    return [
        backend.probabilities(premise=premise, hypothesis=hypothesis)
        for premise, hypothesis in pairs
    ]


def _probabilities_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    probabilities: dict[str, float] = {}
    for row in rows:
        label = str(row.get("label", "")).lower()
        score = float(row.get("score", 0.0))
        probabilities[label] = score
    return probabilities


def _entailment_probability(probabilities: dict[str, float]) -> float:
    for label, probability in probabilities.items():
        normalized = label.lower().replace("_", " ")
        if "entail" in normalized:
            return clamp_score(probability)
    return 0.0


def _split_pipeline_output(output: Any, *, expected: int) -> list[list[dict[str, Any]]]:
    if expected == 0:
        return []
    if isinstance(output, list) and output and isinstance(output[0], list):
        return [
            [row for row in rows if isinstance(row, dict)]
            for rows in output
            if isinstance(rows, list)
        ]
    if expected == 1:
        return [_flatten_pipeline_output(output)]
    return [_flatten_pipeline_output(row) for row in output if isinstance(row, list)]


def _flatten_pipeline_output(output: Any) -> list[dict[str, Any]]:
    if isinstance(output, list):
        return [row for row in output if isinstance(row, dict)]
    if isinstance(output, dict):
        return [output]
    return []
