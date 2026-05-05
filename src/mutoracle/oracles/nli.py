"""Natural-language inference oracle for context-answer faithfulness."""

from __future__ import annotations

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

        output = self._pipeline({"text": premise, "text_pair": hypothesis})
        rows = _flatten_pipeline_output(output)
        probabilities: dict[str, float] = {}
        for row in rows:
            label = str(row.get("label", "")).lower()
            score = float(row.get("score", 0.0))
            probabilities[label] = score
        return probabilities


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
        premise = context_text(run)
        hypothesis = run.answer.strip()
        if not premise or not hypothesis:
            return OracleScore(
                oracle_name=self.name,
                value=0.0,
                metadata={
                    "input_hash": input_hash,
                    "model": self.model_name,
                    "reason": "empty_premise_or_hypothesis",
                },
            )

        probabilities = self._backend.probabilities(
            premise=premise,
            hypothesis=hypothesis,
        )
        entailment_probability = _entailment_probability(probabilities)
        return OracleScore(
            oracle_name=self.name,
            value=entailment_probability,
            metadata={
                "input_hash": input_hash,
                "model": self.model_name,
                "label_probabilities": probabilities,
            },
        )


def _entailment_probability(probabilities: dict[str, float]) -> float:
    for label, probability in probabilities.items():
        normalized = label.lower().replace("_", " ")
        if "entail" in normalized:
            return clamp_score(probability)
    return 0.0


def _flatten_pipeline_output(output: Any) -> list[dict[str, Any]]:
    if isinstance(output, list) and output and isinstance(output[0], list):
        output = output[0]
    if isinstance(output, list):
        return [row for row in output if isinstance(row, dict)]
    if isinstance(output, dict):
        return [output]
    return []
